"""Tests for the v2 real-rotation benchmark variant.

Covers the three mechanically new pieces of the real-rotation variant:

1. the tcp rotation provider (quaternion -> matrix, and the env-reading factory);
2. the v2 oracle path -- ``policy_action_for_condition`` inverts the wrapper's
   ``frame="tool"`` decode exactly when handed the same live rotation; and
3. belief-replica correctness under real rotation -- the pool driver's per
   hypothesis replicas stay bit-faithful to the wrapper's decoder, and the belief
   identifies a tool-frame contract from real-rotation evidence that is invisible
   under the identity placeholder.

Every test is synthetic and CPU-only; no ManiSkill or GPU dependency.
"""

from __future__ import annotations

import torch

from actionshift.adaptation.hypotheses import (
    ExactBeliefDriver,
    HypothesisSimulator,
    identity_rotation,
)
from actionshift.adaptation.response import ResponseModel
from actionshift.benchmarking.ppo_parity import (
    make_tcp_rotation_provider,
    policy_action_for_condition,
)
from actionshift.contracts.transforms import (
    CompleteActionDecoder,
    quaternion_to_rotation_matrix,
)
from actionshift.contracts.types import ActionContract

# A unit quaternion for a 120-degree rotation about (1, 1, 1): a cyclic axis swap,
# clearly non-identity so base and tool frames become distinguishable.
_CYCLIC_QUATERNION = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float64)
_CYCLIC_MATRIX = torch.tensor(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
)


def _contract(*, frame: str = "base", **overrides: object) -> ActionContract:
    values: dict[str, object] = {
        "permutation": tuple(range(6)),
        "sign": (1,) * 6,
        "scale": (1.0,) * 6,
        "target": "delta",
        "frame": frame,
        "lag": 0,
        "gripper_inverted": False,
    }
    values.update(overrides)
    return ActionContract(**values)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 1. Rotation provider.                                                       #
# --------------------------------------------------------------------------- #
def test_quaternion_to_rotation_matrix_known_values() -> None:
    identity = quaternion_to_rotation_matrix(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(identity, torch.eye(3))

    # 90 degrees about z: (x, y, z) -> (-y, x, z).
    root_half = 2.0**-0.5
    z_ninety = quaternion_to_rotation_matrix(
        torch.tensor([root_half, 0.0, 0.0, root_half])
    )
    expected = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    torch.testing.assert_close(z_ninety, expected, atol=1e-6, rtol=1e-6)

    torch.testing.assert_close(
        quaternion_to_rotation_matrix(_CYCLIC_QUATERNION), _CYCLIC_MATRIX
    )


def test_quaternion_to_rotation_matrix_is_orthonormal_and_proper() -> None:
    generator = torch.Generator().manual_seed(20260718)
    quaternion = torch.randn(5, 7, 4, generator=generator, dtype=torch.float64)
    rotation = quaternion_to_rotation_matrix(quaternion)
    assert rotation.shape == (5, 7, 3, 3)
    gram = rotation.transpose(-1, -2) @ rotation
    torch.testing.assert_close(gram, torch.eye(3, dtype=torch.float64).expand_as(gram))
    torch.testing.assert_close(
        torch.linalg.det(rotation), torch.ones(5, 7, dtype=torch.float64)
    )


def test_make_tcp_rotation_provider_reads_env_quaternion() -> None:
    quaternion = torch.stack([_CYCLIC_QUATERNION, torch.tensor([1.0, 0.0, 0.0, 0.0])])

    class _Pose:
        q = quaternion

    class _Tcp:
        pose = _Pose()

    class _Agent:
        tcp = _Tcp()

    class _Unwrapped:
        agent = _Agent()

    class _Env:
        unwrapped = _Unwrapped()

    provider = make_tcp_rotation_provider(_Env())
    rotation = provider()
    assert rotation.shape == (2, 3, 3)
    torch.testing.assert_close(rotation, quaternion_to_rotation_matrix(quaternion))


# --------------------------------------------------------------------------- #
# 2. v2 oracle inversion.                                                      #
# --------------------------------------------------------------------------- #
def test_oracle_path_inverts_tool_frame_decode_under_real_rotation() -> None:
    tool = _contract(
        frame="tool",
        permutation=(1, 0, 2, 4, 5, 3),
        sign=(-1, 1, -1, 1, -1, 1),
        scale=(0.5, 2.0, 1.5, 0.75, 1.25, 0.6),
    )
    rotation = _CYCLIC_MATRIX.expand(2, 3, 3)
    canonical = torch.tensor(
        [
            [0.10, -0.20, 0.30, -0.40, 0.50, -0.60, 0.70],
            [-0.05, 0.15, -0.25, 0.35, -0.45, 0.55, -0.65],
        ],
        dtype=torch.float64,
    )
    raw = policy_action_for_condition(
        canonical,
        "oracle_nonidentity",
        contract=tool,
        tracked_target=torch.zeros((2, 6), dtype=torch.float64),
        ee_rotation=rotation,
    )
    decoder = CompleteActionDecoder(tool, batch_size=2)
    decoded = decoder.step(raw, ee_rotation=rotation)
    torch.testing.assert_close(decoded, canonical)

    # The tool contract under real rotation is genuinely non-degenerate: decoding
    # the oracle-encoded raw against the identity placeholder recovers the WRONG
    # canonical, which is exactly the ceiling a no-adapt policy would hit.
    identity = identity_rotation(2, canonical.device, canonical.dtype)
    wrong = CompleteActionDecoder(tool, batch_size=2).step(raw, ee_rotation=identity)
    assert not torch.allclose(wrong, canonical, atol=1e-3)


def test_oracle_path_default_rotation_matches_identity() -> None:
    tool = _contract(frame="tool")
    canonical = torch.randn(3, 7, dtype=torch.float64)
    default = policy_action_for_condition(
        canonical, "oracle_nonidentity", contract=tool
    )
    explicit_identity = policy_action_for_condition(
        canonical,
        "oracle_nonidentity",
        contract=tool,
        ee_rotation=identity_rotation(3, canonical.device, canonical.dtype),
    )
    torch.testing.assert_close(default, explicit_identity)


# --------------------------------------------------------------------------- #
# 3. Replica correctness and identification under real rotation.              #
# --------------------------------------------------------------------------- #
def test_hypothesis_replicas_match_wrapper_decoder_under_real_rotation() -> None:
    contracts = (
        _contract(frame="base", target="absolute", lag=1),
        _contract(
            frame="tool",
            permutation=(2, 0, 1, 5, 3, 4),
            sign=(1, -1, 1, -1, 1, -1),
            scale=(1.5, 0.5, 2.0, 0.75, 1.25, 0.6),
            lag=2,
        ),
    )
    simulator = HypothesisSimulator(contracts, batch_size=4)
    references = [CompleteActionDecoder(c, batch_size=4) for c in contracts]
    generator = torch.Generator().manual_seed(99)
    quaternion = torch.randn(4, 4, generator=generator, dtype=torch.float64)
    rotation = quaternion_to_rotation_matrix(quaternion).to(torch.float32)

    for step in range(6):
        raw = torch.randn(4, 7, generator=generator, dtype=torch.float32)
        reset = torch.tensor([step == 3, False, step == 3, False])
        predicted = simulator.step(raw, reset_mask=reset, ee_rotation=rotation)
        for index, reference in enumerate(references):
            expected = reference.step(raw, ee_rotation=rotation, reset_mask=reset)
            torch.testing.assert_close(predicted[index], expected)


def test_belief_identifies_tool_frame_only_with_real_rotation() -> None:
    base = _contract(frame="base")
    tool = _contract(frame="tool")
    pool = (base, tool)
    response = ResponseModel(alpha=1.0, sigma=0.02)
    rotation = _CYCLIC_MATRIX.to(torch.float32).expand(3, 3, 3)
    truth = CompleteActionDecoder(tool, batch_size=3)

    real_driver = ExactBeliefDriver(pool, batch_size=3, response=response)
    identity_driver = ExactBeliefDriver(pool, batch_size=3, response=response)
    generator = torch.Generator().manual_seed(7)
    for _ in range(4):
        raw = torch.randn(3, 7, generator=generator) * 0.3
        observed = truth.step(raw, ee_rotation=rotation)[..., :6]
        real_driver.update(raw, observed, ee_rotation=rotation)
        identity_driver.update(raw, observed)

    tool_index = pool.index(tool)
    # Real rotation resolves the frame: every environment's MAP is the tool contract.
    assert torch.all(real_driver.map_indices() == tool_index)
    real_probabilities = real_driver.log_probabilities.exp()
    assert torch.all(real_probabilities[:, tool_index] > 0.99)

    # Under the identity placeholder the two hypotheses are observationally
    # identical up to floating-point rounding, so essentially no evidence
    # accumulates and the belief stays at the uniform prior (never concentrates).
    identity_probabilities = identity_driver.log_probabilities.exp()
    assert torch.all((identity_probabilities - 0.5).abs() < 1e-3)


def test_map_encode_round_trips_through_wrapper_under_real_rotation() -> None:
    base = _contract(frame="base")
    tool = _contract(frame="tool")
    pool = (base, tool)
    response = ResponseModel(alpha=1.0, sigma=0.02)
    rotation = _CYCLIC_MATRIX.to(torch.float32).expand(3, 3, 3)
    truth = CompleteActionDecoder(tool, batch_size=3)

    driver = ExactBeliefDriver(pool, batch_size=3, response=response)
    generator = torch.Generator().manual_seed(11)
    for _ in range(4):
        raw = torch.randn(3, 7, generator=generator) * 0.3
        observed = truth.step(raw, ee_rotation=rotation)[..., :6]
        driver.update(raw, observed, ee_rotation=rotation)

    # With the tool contract identified, the raw command produced by ``map_encode``
    # must decode back to the intended canonical action through the wrapper's tool
    # decode against the same live rotation.
    canonical = torch.randn(3, 7, generator=generator) * 0.2
    encoded = driver.map_encode(canonical, ee_rotation=rotation)
    decoded = CompleteActionDecoder(tool, batch_size=3).step(encoded, ee_rotation=rotation)
    torch.testing.assert_close(decoded, canonical, atol=1e-5, rtol=1e-5)
