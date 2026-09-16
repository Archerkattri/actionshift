"""Frozen, simulator-independent compound-shift challenge protocol.

The runner owns private contracts and exposes only canonical policy commands,
executed responses, rewards, and the public episode schedule to adapters.  It is
small enough for a CPU smoke test, but the adapter ABI is the same boundary a
simulator integration can implement without changing ActionShift internals.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

import torch

from actionshift.contracts.compound import CompoundShiftEpisode, generate_compound_episode
from actionshift.contracts.transforms import ActionLag, decode_pose, encode_pose

CHALLENGE_VERSION = "compound-challenge-v1"
CHALLENGE_CASES: tuple[tuple[str, bool], ...] = (
    ("static", False),
    ("abrupt", False),
    ("gradual", False),
    ("recurring", False),
    ("abrupt", True),
)
DEFAULT_CHALLENGE_SEEDS = (20260718, 20260719, 20260720)


@dataclass(frozen=True, slots=True)
class MethodBudget:
    """Per-episode envelope shared by every challenge method."""

    episode_steps: int
    max_probe_steps: int
    max_action_linf: float = 1.0
    max_probe_linf: float = 0.5

    def __post_init__(self) -> None:
        if self.episode_steps < 8:
            raise ValueError("episode_steps must be at least eight")
        if not 0 <= self.max_probe_steps <= self.episode_steps:
            raise ValueError("max_probe_steps must fit inside the episode")
        if not 0 < self.max_probe_linf <= self.max_action_linf:
            raise ValueError("probe and action bounds must be positive and ordered")


@dataclass(frozen=True, slots=True)
class ChallengeObservation:
    step: int
    canonical_action: tuple[float, ...]
    previous_raw_action: tuple[float, ...]
    observed_response: tuple[float, ...]
    reward: float
    shift_boundary: bool


@dataclass(frozen=True, slots=True)
class ChallengeDecision:
    raw_action: tuple[float, ...]
    probe: bool = False


@dataclass(frozen=True, slots=True)
class ChallengeTransition:
    step: int
    canonical_action: tuple[float, ...]
    raw_action: tuple[float, ...]
    observed_response: tuple[float, ...]
    reward: float
    probe: bool


class ChallengeAdapter(Protocol):
    """Public adapter ABI.  No method receives an ``ActionContract``."""

    def reset(self, *, manifest: Mapping[str, object], budget: MethodBudget) -> None: ...

    def act(self, observation: ChallengeObservation) -> ChallengeDecision: ...

    def observe(self, transition: ChallengeTransition) -> None: ...


AdapterFactory = Callable[..., ChallengeAdapter]


@dataclass(frozen=True, slots=True)
class ChallengeMethodSpec:
    name: str
    max_probe_steps: int
    privileges: tuple[str, ...] = ()
    claim_label: str = "local_matched_budget_implementation"


def challenge_method_specs() -> tuple[ChallengeMethodSpec, ...]:
    """Method envelopes corresponding exactly to the frozen headline matrix."""
    probe = ("bounded_active_probe",)
    return (
        ChallengeMethodSpec("oracle", 0, ("true_contract",), "privileged_ceiling"),
        ChallengeMethodSpec("no_adapt", 0),
        ChallengeMethodSpec("domain_randomized", 0),
        ChallengeMethodSpec("recurrent", 0),
        ChallengeMethodSpec("osi", 0),
        ChallengeMethodSpec("rma", 0, ("teacher_contract_during_training",)),
        ChallengeMethodSpec("random_probes", 6, probe),
        ChallengeMethodSpec("fixed_probes", 6, probe),
        ChallengeMethodSpec("dualabi", 6, probe, "experimental_method"),
        ChallengeMethodSpec("dualabi_entropy", 6, probe, "experimental_ablation"),
    )


class IdentityChallengeAdapter:
    """Reference no-adaptation adapter and minimal external-ABI example."""

    def reset(self, *, manifest: Mapping[str, object], budget: MethodBudget) -> None:
        del manifest, budget

    def act(self, observation: ChallengeObservation) -> ChallengeDecision:
        return ChallengeDecision(observation.canonical_action)

    def observe(self, transition: ChallengeTransition) -> None:
        del transition


def identity_adapter_factory(*, seed: int, budget: MethodBudget) -> ChallengeAdapter:
    """Factory addressable as ``actionshift.evaluation.challenge:identity_adapter_factory``."""
    del seed, budget
    return IdentityChallengeAdapter()


def load_adapter_factory(reference: str) -> AdapterFactory:
    """Load an external ``module:factory`` without editing simulator or package code."""
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("adapter must use module:factory syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise ValueError(f"adapter factory is not callable: {reference}")
    return cast(AdapterFactory, factory)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_digest(value: Mapping[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_challenge_manifest(
    repository_root: Path,
    *,
    seeds: Sequence[int] = DEFAULT_CHALLENGE_SEEDS,
    steps: int = 64,
) -> dict[str, object]:
    """Build the deterministic public protocol and pin every source it depends on."""
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonempty and nonnegative")
    source_paths = (
        Path("src/actionshift/contracts/compound.py"),
        Path("src/actionshift/contracts/transforms.py"),
        Path("src/actionshift/evaluation/challenge.py"),
        Path("configs/evaluation/headline.yaml"),
        Path("configs/sprint/sources.yaml"),
        Path("configs/split/manifests/seen.json"),
        Path("configs/split/manifests/unseen_value.json"),
        Path("configs/split/manifests/unseen_composition.json"),
        Path("configs/split/manifests/long_lag.json"),
        Path("configs/split/manifests/task_transfer.json"),
    )
    sources = []
    for relative in source_paths:
        absolute = repository_root / relative
        if not absolute.is_file():
            raise FileNotFoundError(absolute)
        sources.append({"path": relative.as_posix(), "sha256": _sha256(absolute)})
    budget = MethodBudget(steps, 6)
    public_episodes = [
        generate_compound_episode(kind, seed=seed, steps=steps, outside_grammar=outside)
        .public_manifest()
        for seed in seeds
        for kind, outside in CHALLENGE_CASES
    ]
    protocol: dict[str, object] = {
        "schema_version": "1.0",
        "challenge_version": CHALLENGE_VERSION,
        "status": "FROZEN",
        "claim_boundary": "CPU protocol smoke; not a ManiSkill performance result",
        "seeds": list(seeds),
        "cases": [
            {"shift_kind": kind, "outside_grammar": outside}
            for kind, outside in CHALLENGE_CASES
        ],
        "budget": asdict(budget),
        "methods": [
            {**asdict(spec), "privileges": list(spec.privileges)}
            for spec in challenge_method_specs()
        ],
        "episodes": public_episodes,
        "sources": sources,
    }
    return {**protocol, "protocol_sha256": _protocol_digest(protocol)}


def verify_challenge_manifest(manifest: Mapping[str, object]) -> None:
    supplied = manifest.get("protocol_sha256")
    if not isinstance(supplied, str):
        raise ValueError("manifest lacks protocol_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "protocol_sha256"}
    if _protocol_digest(unsigned) != supplied:
        raise ValueError("challenge manifest digest mismatch")


def write_challenge_manifest(path: Path, repository_root: Path) -> dict[str, object]:
    manifest = build_challenge_manifest(repository_root)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("refusing to overwrite a different frozen challenge manifest")
        return manifest
    _atomic_json(path, manifest)
    return manifest


def _canonical_action(seed: int, step: int, dimension: int) -> torch.Tensor:
    phase = 0.31 * (step + 1) + (seed % 97) * 0.017
    values = [0.24 * math.sin(phase + index * 0.83) for index in range(dimension)]
    return torch.tensor(values, dtype=torch.float64)


def _segment_index(episode: CompoundShiftEpisode, step: int) -> int:
    for index, segment in enumerate(episode.segments):
        if segment.start <= step < segment.stop:
            return index
    raise RuntimeError("step is outside episode segments")


def _validated_action(
    decision: ChallengeDecision, budget: MethodBudget, dimension: int
) -> torch.Tensor:
    if not isinstance(decision, ChallengeDecision):
        raise TypeError("adapter.act must return ChallengeDecision")
    if len(decision.raw_action) != dimension:
        raise ValueError("adapter action dimension mismatch")
    action = torch.tensor(decision.raw_action, dtype=torch.float64)
    if not torch.isfinite(action).all():
        raise ValueError("adapter action must be finite")
    bound = budget.max_probe_linf if decision.probe else budget.max_action_linf
    if float(action.abs().max()) > bound + 1e-12:
        raise ValueError("adapter action exceeds its declared bound")
    return action


def _run_episode(
    episode: CompoundShiftEpisode,
    *,
    method: str,
    factory: AdapterFactory | None,
    max_probe_steps: int,
) -> dict[str, object]:
    dimension = len(episode.contract_at(0).permutation)
    budget = MethodBudget(episode.steps, max_probe_steps)
    adapter: ChallengeAdapter | None = None
    if method != "oracle":
        selected_factory = factory or identity_adapter_factory
        adapter = selected_factory(seed=episode.seed, budget=budget)
        for member in ("reset", "act", "observe"):
            if not callable(getattr(adapter, member, None)):
                raise TypeError(f"adapter is missing callable {member}")
        adapter.reset(manifest=episode.public_manifest(), budget=budget)

    previous_raw = torch.zeros(dimension, dtype=torch.float64)
    previous_response = torch.zeros(dimension, dtype=torch.float64)
    previous_reward = 0.0
    lag: ActionLag | None = None
    active_segment = -1
    probes = 0
    squared_errors: list[float] = []
    scored_errors: list[float] = []
    action_cost = 0.0

    for step in range(episode.steps):
        segment_index = _segment_index(episode, step)
        segment = episode.segments[segment_index]
        boundary = segment_index != active_segment
        if boundary:
            active_segment = segment_index
            lag = ActionLag(steps=segment.contract.lag)
        assert lag is not None
        canonical = _canonical_action(episode.seed, step, dimension)
        if method == "oracle":
            future_step = step + segment.contract.lag
            future = (
                _canonical_action(episode.seed, future_step, dimension)
                if future_step < segment.stop
                else torch.zeros_like(canonical)
            )
            raw = encode_pose(future, segment.contract)
            decision = ChallengeDecision(tuple(float(value) for value in raw), False)
        else:
            assert adapter is not None
            observation = ChallengeObservation(
                step=step,
                canonical_action=tuple(float(value) for value in canonical),
                previous_raw_action=tuple(float(value) for value in previous_raw),
                observed_response=tuple(float(value) for value in previous_response),
                reward=previous_reward,
                shift_boundary=boundary,
            )
            decision = adapter.act(observation)
        raw = _validated_action(decision, budget, dimension)
        probes += int(decision.probe)
        if probes > budget.max_probe_steps:
            raise ValueError("adapter exceeded its probe-step budget")
        decoded = decode_pose(raw, segment.contract)
        response = lag.step(decoded)
        error = float(torch.mean((response - canonical) ** 2))
        squared_errors.append(error)
        if step >= segment.start + segment.contract.lag:
            scored_errors.append(error)
        reward = -error
        action_cost += float(torch.linalg.vector_norm(raw))
        transition = ChallengeTransition(
            step=step,
            canonical_action=tuple(float(value) for value in canonical),
            raw_action=tuple(float(value) for value in raw),
            observed_response=tuple(float(value) for value in response),
            reward=reward,
            probe=decision.probe,
        )
        if adapter is not None:
            adapter.observe(transition)
        previous_raw, previous_response, previous_reward = raw, response, reward

    scored_mse = sum(scored_errors) / len(scored_errors)
    return {
        "episode_id": episode.episode_id,
        "seed": episode.seed,
        "shift_kind": episode.shift_kind,
        "outside_grammar": episode.outside_grammar,
        "steps": episode.steps,
        "probe_steps": probes,
        "mean_squared_error": sum(squared_errors) / len(squared_errors),
        "post_lag_mean_squared_error": scored_mse,
        "post_lag_success": math.sqrt(scored_mse) <= 0.05,
        "cumulative_action_cost": action_cost,
    }


def run_challenge(
    *,
    method: str,
    factory: AdapterFactory | None = None,
    seeds: Sequence[int] = DEFAULT_CHALLENGE_SEEDS,
    steps: int = 64,
) -> dict[str, object]:
    specs = {spec.name: spec for spec in challenge_method_specs()}
    if method == "external":
        if factory is None:
            raise ValueError("external method requires an adapter factory")
        max_probes = 6
        privileges: tuple[str, ...] = ()
    elif method in specs:
        if method not in {"oracle", "no_adapt"} and factory is None:
            raise ValueError(f"{method} requires an explicit adapter factory")
        max_probes = specs[method].max_probe_steps
        privileges = specs[method].privileges
    else:
        raise ValueError(f"unknown challenge method: {method}")
    records = [
        _run_episode(
            generate_compound_episode(kind, seed=seed, steps=steps, outside_grammar=outside),
            method=method,
            factory=factory,
            max_probe_steps=max_probes,
        )
        for seed in seeds
        for kind, outside in CHALLENGE_CASES
    ]
    successes = 0
    post_lag_mse_total = 0.0
    for record in records:
        success = record["post_lag_success"]
        post_lag_mse = record["post_lag_mean_squared_error"]
        if not isinstance(success, bool):
            raise TypeError("challenge episode success must be boolean")
        if isinstance(post_lag_mse, bool) or not isinstance(post_lag_mse, (int, float)):
            raise TypeError("challenge episode MSE must be numeric")
        successes += int(success)
        post_lag_mse_total += float(post_lag_mse)
    return {
        "schema_version": "1.0",
        "challenge_version": CHALLENGE_VERSION,
        "claim_boundary": "CPU protocol smoke; not a ManiSkill performance result",
        "method": method,
        "privileges": list(privileges),
        "seeds": list(seeds),
        "episode_count": len(records),
        "successes": successes,
        "mean_post_lag_mse": post_lag_mse_total / len(records),
        "episodes": records,
    }


def run_reference_smoke(
    path: Path,
    *,
    seeds: Sequence[int] = DEFAULT_CHALLENGE_SEEDS,
    steps: int = 64,
    external_reference: str | None = None,
) -> dict[str, object]:
    repository_root = Path(__file__).resolve().parents[3]
    protocol = build_challenge_manifest(repository_root, seeds=seeds, steps=steps)
    methods = [
        run_challenge(method="oracle", seeds=seeds, steps=steps),
        run_challenge(method="no_adapt", seeds=seeds, steps=steps),
    ]
    if external_reference is not None:
        methods.append(
            run_challenge(
                method="external",
                factory=load_adapter_factory(external_reference),
                seeds=seeds,
                steps=steps,
            )
        )
    report: dict[str, object] = {
        "schema_version": "1.0",
        "challenge_version": CHALLENGE_VERSION,
        "protocol_sha256": protocol["protocol_sha256"],
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "methods": methods,
    }
    _atomic_json(path, report)
    return report


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


__all__ = [
    "CHALLENGE_CASES",
    "CHALLENGE_VERSION",
    "ChallengeAdapter",
    "ChallengeDecision",
    "ChallengeObservation",
    "ChallengeTransition",
    "MethodBudget",
    "build_challenge_manifest",
    "challenge_method_specs",
    "identity_adapter_factory",
    "load_adapter_factory",
    "run_challenge",
    "run_reference_smoke",
    "verify_challenge_manifest",
    "write_challenge_manifest",
]
