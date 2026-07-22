"""Matched-budget Gate 1 method registry and pilot planning."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True, slots=True)
class FrozenProtocol:
    backbone_sha256: str
    actor_parameters: int
    critic_parameters: int
    history_steps: int
    observation: str
    environment_steps: int
    updates: int
    train_split_sha256: str
    evaluation_split_sha256: str
    probe_budget: int

    def __post_init__(self) -> None:
        for name in ("backbone_sha256", "train_split_sha256", "evaluation_split_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        for name in (
            "actor_parameters",
            "critic_parameters",
            "history_steps",
            "environment_steps",
            "updates",
            "probe_budget",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.observation:
            raise ValueError("observation must be nonempty")


@dataclass(frozen=True, slots=True)
class MethodSpec:
    name: str
    runnable: bool
    implementation: str
    claim_label: str
    privileges: frozenset[str] = frozenset()
    official_source: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.implementation or not self.claim_label:
            raise ValueError("method fields must be nonempty")
        if not self.runnable and not self.reason:
            raise ValueError("inapplicable methods require a reason")
        if self.claim_label == "faithful_external_reproduction" and not self.official_source:
            raise ValueError("external reproductions require an official source record")


@dataclass(frozen=True, slots=True)
class AdaptationJob:
    task: str
    method: str
    split: str
    seed: int
    protocol: FrozenProtocol
    privileges: frozenset[str]

    @property
    def job_id(self) -> str:
        value = asdict(self)
        value["privileges"] = sorted(self.privileges)
        payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class AdaptationPlan:
    jobs: tuple[AdaptationJob, ...]
    exclusions: Mapping[str, str]


def method_registry() -> dict[str, MethodSpec]:
    """Return stable labels without presenting local analogues as official reproductions."""
    local = "local_matched_budget_implementation"

    def component(
        name: str,
        implementation: str,
        claim_label: str = local,
        privileges: frozenset[str] = frozenset(),
    ) -> MethodSpec:
        return MethodSpec(
            name,
            False,
            implementation,
            claim_label,
            privileges,
            reason=(
                "Component exists, but no trained matched-budget adapter checkpoint "
                "and end-to-end runner is available."
            ),
        )

    return {
        "oracle": MethodSpec(
            "oracle",
            True,
            "actionshift.methods.oracle",
            "privileged_ceiling",
            frozenset({"true_contract"}),
        ),
        "no_adapt": MethodSpec("no_adapt", True, "actionshift.methods.no_adapt", local),
        "domain_randomized": component(
            "domain_randomized", "actionshift.methods.no_adapt"
        ),
        "recurrent": component("recurrent", "actionshift.methods.recurrent"),
        "up_osi": component(
            "up_osi", "actionshift.methods.osi", "UP-OSI-style local baseline"
        ),
        "rma": component(
            "rma",
            "actionshift.methods.osi",
            "RMA-style local baseline",
            frozenset({"teacher_contract_during_training"}),
        ),
        "fixed_probes": component(
            "fixed_probes",
            "actionshift.methods.dualabi",
            local,
            frozenset({"bounded_active_probe"}),
        ),
        "random_probes": component(
            "random_probes",
            "actionshift.methods.dualabi",
            local,
            frozenset({"bounded_active_probe"}),
        ),
        "posterior_only": component("posterior_only", "actionshift.methods.dualabi"),
        "entropy": component(
            "entropy",
            "actionshift.methods.dualabi",
            local,
            frozenset({"bounded_active_probe"}),
        ),
        "exact_belief": component("exact_belief", "actionshift.belief.exact"),
        "dualabi": component(
            "dualabi",
            "actionshift.methods.dualabi",
            "experimental_method",
            frozenset({"bounded_active_probe"}),
        ),
        "transformer_icl": MethodSpec(
            "transformer_icl",
            False,
            "unavailable",
            "conditional_external_baseline",
            reason=(
                "No faithful runnable SPACE_GLAM-style or transformer ICL "
                "implementation is pinned. Vintix (dunnolab/vintix, ICML 2025, "
                "commit 2b10276, Apache-2.0) adjudicated and excluded: its "
                "dimension-grouped encoder/decoder heads have no (obs=42, act=7) "
                "group and cannot accept ActionShift's obs/action dims without new "
                "heads, its per-task normalization and Algorithm-Distillation "
                "in-context competence are tied to MuJoCo/Meta-World/Bi-DexHands/"
                "Industrial-Benchmark (no ManiSkill exposure), and it adapts to "
                "KNOWN training tasks, not a hidden action-ABI contract; a faithful "
                "application would require new heads plus full retraining on "
                "ManiSkill learning histories, which is not a reproduction. See "
                "reports/transformer_icl_adjudication.md."
            ),
        ),
    }


def plan_gate1_pilots(
    passing_backbones: Mapping[str, FrozenProtocol],
    *,
    seed: int,
    splits: Sequence[str],
    methods: Sequence[str] | None = None,
) -> AdaptationPlan:
    if seed < 0 or not splits or any(not split for split in splits):
        raise ValueError("seed must be nonnegative and splits must be nonempty")
    registry = method_registry()
    selected = tuple(methods) if methods is not None else tuple(registry)
    unknown = sorted(set(selected) - set(registry))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")
    exclusions = {
        name: registry[name].reason or "inapplicable"
        for name in selected
        if not registry[name].runnable
    }
    jobs = tuple(
        AdaptationJob(task, method, split, seed, protocol, registry[method].privileges)
        for task, protocol in sorted(passing_backbones.items())
        for split in splits
        for method in selected
        if registry[method].runnable
    )
    return AdaptationPlan(jobs, exclusions)


def promote_seeds(
    pilots: Iterable[AdaptationJob], additional_seeds: Sequence[int]
) -> tuple[AdaptationJob, ...]:
    pilot_jobs = tuple(pilots)
    seeds = sorted({job.seed for job in pilot_jobs} | set(additional_seeds))
    if any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonnegative")
    unique: dict[str, AdaptationJob] = {}
    for job in pilot_jobs:
        for seed in seeds:
            promoted = replace(job, seed=seed)
            unique[promoted.job_id] = promoted
    return tuple(unique.values())


def superiority_eligible(seeds: Iterable[int], *, require: bool = False) -> bool:
    eligible = len(set(seeds)) >= 3
    if require and not eligible:
        raise ValueError("superiority requires at least three matched seeds")
    return eligible
