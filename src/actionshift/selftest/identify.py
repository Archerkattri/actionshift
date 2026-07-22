"""Run the bounded probe phase and read off an identification result.

The probe phase reuses the exact adaptation stack from
``reports/adaptation_tournament.md``: a bounded ``ProbingBeliefAdapter`` (fixed or
entropy schedule) folds each probe response into an ``ExactBeliefDriver`` over the
declared pool. After ``budget`` steps this module summarizes the belief into an
:class:`IdentificationResult` -- the MAP wiring, per-field marginal posterior mass,
and a misspecification statistic (the MAP hypothesis's chi-square per degree of
freedom against the calibrated noise scale).

Two probe environments share one ``ProbeEnvironment`` protocol: a bit-faithful
synthetic stand-in for the hidden wrapper (CPU, no ManiSkill) used by demos and
tests, and a real ManiSkill environment used by ``--real`` runs.
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from actionshift.adaptation.hypotheses import ExactBeliefDriver, HypothesisSimulator
from actionshift.adaptation.probes import ProbeStrategy, ProbingBeliefAdapter
from actionshift.adaptation.response import ResponseModel
from actionshift.contracts.transforms import CompleteActionDecoder
from actionshift.contracts.types import ActionContract
from actionshift.selftest.diffs import POSE_FIELDS
from actionshift.selftest.verdict import IdentificationResult

_ACTION_WIDTH = 7


class ProbeEnvironment(Protocol):
    """A hidden-contract environment the probe phase can drive step by step."""

    batch_size: int
    channels: int

    def step(self, raw_action: Tensor) -> Tensor:
        """Apply one raw action; return the observed response (batch, channels)."""
        ...

    def close(self) -> None:
        ...


class SyntheticProbeEnvironment:
    """Bit-faithful CPU stand-in for the hidden wrapper (decode-then-lag).

    Uses the very ``CompleteActionDecoder`` the real ``HiddenContractWrapper`` uses,
    so the decode pipeline the probe identifies is identical; only the physics
    simulator is replaced by the calibrated response model. Identity rotation.
    """

    def __init__(
        self,
        contract: ActionContract,
        *,
        batch_size: int,
        response: ResponseModel,
        seed: int = 0,
        noise: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._decoder = CompleteActionDecoder(contract, batch_size=batch_size)
        self.batch_size = batch_size
        self.channels = 7 if response.has_gripper else 6
        self._response = response
        self._noise = noise
        self._generator = torch.Generator().manual_seed(seed)
        self._sigma = _sigma_tensor(response)

    def step(self, raw_action: Tensor) -> Tensor:
        rotation = torch.eye(3, dtype=raw_action.dtype).expand(self.batch_size, 3, 3)
        canonical = self._decoder.step(raw_action, ee_rotation=rotation)
        expected = self._response.expected(canonical)
        if self._noise:
            noise = torch.randn(
                expected.shape, generator=self._generator, dtype=expected.dtype
            )
            expected = expected + noise * self._sigma
        if self.channels == 6:
            return expected
        gripper = canonical[..., 6:7]
        if self._response.gripper_alpha is not None:
            gripper = gripper * self._response.gripper_alpha
        return torch.cat((expected, gripper), dim=-1)

    def close(self) -> None:
        return None


def _sigma_tensor(response: ResponseModel) -> Tensor:
    sigma = response.sigma
    if isinstance(sigma, (int, float)):
        return torch.full((response.channels,), float(sigma))
    return torch.tensor(tuple(float(value) for value in sigma))


def _pool_marginals(
    pool: tuple[ActionContract, ...],
    posterior: Tensor,
    map_contract: ActionContract,
) -> dict[str, float]:
    """Marginal posterior mass on the MAP wiring's value, per observable field."""
    confidence: dict[str, float] = {}
    for field in POSE_FIELDS:
        target = getattr(map_contract, field)
        mass = sum(
            float(posterior[index])
            for index, contract in enumerate(pool)
            if getattr(contract, field) == target
        )
        confidence[field] = mass
    return confidence


def _fit_ratio(
    map_contract: ActionContract,
    raws: list[Tensor],
    observed: list[Tensor],
    response: ResponseModel,
    *,
    batch_size: int,
) -> float:
    """Chi-square per DoF of the MAP hypothesis against the calibrated noise.

    Replays the probe raw actions through a single-hypothesis replica (identity
    rotation, no mid-phase reset) and standardizes the residual by ``sigma``. A
    value near 1 means the wiring explains the responses at the calibrated noise
    scale; a large value means even the best hypothesis fits poorly.
    """
    simulator = HypothesisSimulator((map_contract,), batch_size=batch_size)
    sigma = _sigma_tensor(response)
    total_square = 0.0
    total_terms = 0
    for raw, obs in zip(raws, observed, strict=True):
        predicted = simulator.step(raw)[0]
        expected = response.expected(predicted)
        residual = (obs[..., : response.channels] - expected) / sigma
        total_square += float(residual.square().sum())
        total_terms += residual.numel()
    return total_square / total_terms if total_terms else 0.0


def identify_contract(
    environment: ProbeEnvironment,
    pool: tuple[ActionContract, ...],
    response: ResponseModel,
    *,
    strategy: ProbeStrategy = "entropy",
    budget: int = 6,
    amplitude: float = 0.5,
    seed: int = 0,
) -> IdentificationResult:
    """Drive the bounded probe phase and summarize the belief."""
    if strategy not in ("fixed", "entropy"):
        raise ValueError("self-test probing supports 'fixed' or 'entropy' strategies")
    if budget <= 0:
        raise ValueError("budget must be positive")
    if environment.batch_size <= 0:
        raise ValueError("environment batch_size must be positive")

    driver = ExactBeliefDriver(pool, batch_size=environment.batch_size, response=response)
    adapter = ProbingBeliefAdapter(
        driver, strategy=strategy, budget=budget, amplitude=amplitude, seed=seed
    )
    zero_intent = torch.zeros((environment.batch_size, _ACTION_WIDTH))

    raws: list[Tensor] = []
    observed: list[Tensor] = []
    probe_steps = torch.zeros(environment.batch_size)
    displacement = torch.zeros(environment.batch_size)
    for _ in range(budget):
        raw = adapter.encode(zero_intent)
        response_observation = environment.step(raw)
        adapter.observe(raw, response_observation[..., :6])
        raws.append(raw.detach().clone())
        observed.append(response_observation.detach().clone())
        mask = adapter.last_probe_mask
        if mask is not None:
            probing = mask.detach().reshape(-1).to(torch.float32)
            probe_steps += probing
            displacement += probing * response_observation[..., :3].norm(dim=-1)

    posterior = driver.log_probabilities.exp().mean(dim=0)
    map_index = int(posterior.argmax())
    map_contract = pool[map_index]
    field_confidence = _pool_marginals(pool, posterior, map_contract)
    fit_ratio = _fit_ratio(
        map_contract, raws, observed, response, batch_size=environment.batch_size
    )
    return IdentificationResult(
        map_contract=map_contract,
        pool_posterior=tuple(float(value) for value in posterior),
        field_confidence=field_confidence,
        fit_ratio=fit_ratio,
        probe_steps=float(probe_steps.mean()),
        probe_displacement=float(displacement.mean()),
        strategy=strategy,
    )
