from __future__ import annotations

import warnings

import pytest
import torch

from actionshift.contracts.transforms import (
    ActionLag,
    CompleteActionDecoder,
    decode_complete_action,
    decode_pose,
    encode_complete_action,
    encode_pose,
)
from actionshift.contracts.types import ActionContract

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    CUDA_AVAILABLE = torch.cuda.is_available()


def contract(*, lag: int = 0) -> ActionContract:
    return ActionContract(
        permutation=(1, 0),
        sign=(1, -1),
        scale=(0.5, 2.0),
        target="delta",
        frame="base",
        lag=lag,
        gripper_inverted=False,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_decode_then_encode_round_trips(dtype: torch.dtype) -> None:
    raw = torch.tensor([[0.25, -0.8], [0.5, 0.1]], dtype=dtype)

    decoded = decode_pose(raw, contract())
    reconstructed = encode_pose(decoded, contract())

    tolerance = 1e-5 if dtype == torch.float32 else 1e-6
    torch.testing.assert_close(reconstructed, raw, rtol=tolerance, atol=tolerance)
    assert decoded.dtype == raw.dtype
    assert decoded.device == raw.device


def test_decode_uses_permutation_then_sign_then_scale() -> None:
    raw = torch.tensor([[2.0, 3.0]])

    decoded = decode_pose(raw, contract())

    torch.testing.assert_close(decoded, torch.tensor([[1.5, -4.0]]))


def test_permutation_and_scale_are_noncommuting() -> None:
    raw = torch.tensor([[2.0, 3.0]])
    decoded = decode_pose(raw, contract())
    scale_before_permutation = (raw * torch.tensor([0.5, 2.0]))[:, [1, 0]]
    scale_before_permutation *= torch.tensor([1.0, -1.0])

    assert not torch.equal(decoded, scale_before_permutation)


def test_decode_clips_only_after_contract_transform() -> None:
    raw = torch.tensor([[2.0, 3.0]])

    decoded = decode_pose(raw, contract(), lower=-1.0, upper=1.0)

    torch.testing.assert_close(decoded, torch.tensor([[1.0, -1.0]]))


def test_decode_preserves_gradient_flow() -> None:
    raw = torch.tensor([[2.0, 3.0]], requires_grad=True)

    decode_pose(raw, contract()).sum().backward()

    torch.testing.assert_close(raw.grad, torch.tensor([[-2.0, 0.5]]))


def test_lag_buffer_emits_neutral_actions_then_delayed_actions() -> None:
    lag = ActionLag(steps=2)
    first = torch.tensor([[1.0, 10.0]])
    second = torch.tensor([[2.0, 20.0]])
    third = torch.tensor([[3.0, 30.0]])

    outputs = [lag.step(first), lag.step(second), lag.step(third)]

    torch.testing.assert_close(outputs[0], torch.zeros_like(first))
    torch.testing.assert_close(outputs[1], torch.zeros_like(first))
    torch.testing.assert_close(outputs[2], first)


def test_lag_zero_returns_the_same_tensor() -> None:
    lag = ActionLag(steps=0)
    action = torch.tensor([[1.0, 2.0]], requires_grad=True)

    output = lag.step(action)

    assert output is action


def test_reset_mask_clears_only_selected_vectorized_environment() -> None:
    lag = ActionLag(steps=1)
    first = torch.tensor([[1.0], [10.0]])
    second = torch.tensor([[2.0], [20.0]])

    lag.step(first)
    output = lag.step(second, reset_mask=torch.tensor([True, False]))

    torch.testing.assert_close(output, torch.tensor([[0.0], [10.0]]))


def test_lag_rejects_shape_changes_without_explicit_reset() -> None:
    lag = ActionLag(steps=1)
    lag.step(torch.zeros(2, 3))

    with pytest.raises(ValueError, match="action shape changed"):
        lag.step(torch.zeros(1, 3))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is unavailable")
def test_decode_and_lag_preserve_cuda_device() -> None:
    raw = torch.tensor([[2.0, 3.0]], device="cuda")
    lag = ActionLag(steps=1)

    decoded = decode_pose(raw, contract())
    output = lag.step(decoded)

    assert decoded.device.type == "cuda"
    assert output.device.type == "cuda"


def full_contract(**overrides: object) -> ActionContract:
    values: dict[str, object] = {
        "permutation": tuple(range(6)),
        "sign": (1,) * 6,
        "scale": (1.0,) * 6,
        "target": "delta",
        "frame": "base",
        "lag": 0,
        "gripper_inverted": False,
    }
    values.update(overrides)
    return ActionContract(**values)  # type: ignore[arg-type]


def test_complete_decode_rotates_tool_twist_and_keeps_named_gripper() -> None:
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    raw = torch.tensor([1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.4], dtype=torch.float64)

    canonical, target = decode_complete_action(
        raw,
        full_contract(frame="tool"),
        ee_rotation=rotation,
        tracked_target=torch.zeros(6, dtype=torch.float64),
    )

    torch.testing.assert_close(
        canonical,
        torch.tensor([0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.4], dtype=torch.float64),
    )
    torch.testing.assert_close(target, canonical[:6])


def test_absolute_target_uses_tracked_target_and_workspace_clip() -> None:
    raw = torch.tensor([2.0, -2.0, 0.0, 0.0, 0.0, 0.0, -0.2])
    canonical, target = decode_complete_action(
        raw,
        full_contract(target="absolute", gripper_inverted=True),
        ee_rotation=torch.eye(3),
        tracked_target=torch.tensor([0.25, -0.25, 0.0, 0.0, 0.0, 0.0]),
        workspace_lower=torch.tensor([-0.5, -0.5, -1.0]),
        workspace_upper=torch.tensor([0.5, 0.5, 1.0]),
    )

    torch.testing.assert_close(canonical[:3], torch.tensor([0.25, -0.25, 0.0]))
    torch.testing.assert_close(target[:3], torch.tensor([0.5, -0.5, 0.0]))
    assert canonical[-1].item() == pytest.approx(0.2)


def test_complete_decoder_resets_absolute_target_and_lag_per_environment() -> None:
    decoder = CompleteActionDecoder(full_contract(target="absolute", lag=1), batch_size=2)
    rotation = torch.eye(3).expand(2, 3, 3)
    first = torch.tensor(
        [
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    torch.testing.assert_close(decoder.step(first, ee_rotation=rotation), torch.zeros_like(first))

    second = decoder.step(first, ee_rotation=rotation, reset_mask=torch.tensor([True, False]))

    torch.testing.assert_close(second[0], torch.zeros(7))
    torch.testing.assert_close(second[1, :6], first[1, :6])


def test_complete_decode_rejects_non_rotation_matrix() -> None:
    with pytest.raises(ValueError, match="rotation"):
        decode_complete_action(
            torch.zeros(7),
            full_contract(frame="tool"),
            ee_rotation=torch.zeros(3, 3),
            tracked_target=torch.zeros(6),
        )


def test_complete_oracle_round_trip_for_100_random_compositions() -> None:
    generator = torch.Generator().manual_seed(20260718)
    for index in range(100):
        permutation = tuple(int(value) for value in torch.randperm(6, generator=generator))
        sign = tuple(
            int(value) for value in (torch.randint(0, 2, (6,), generator=generator) * 2 - 1)
        )
        scale = tuple(
            float(value) for value in torch.rand(6, generator=generator, dtype=torch.float64) + 0.25
        )
        candidate = full_contract(
            permutation=permutation,
            sign=sign,
            scale=scale,
            target="absolute" if index % 2 else "delta",
            frame="tool" if index % 3 else "base",
            gripper_inverted=bool(index % 5 == 0),
        )
        angle = torch.rand((), generator=generator, dtype=torch.float64) * 6.0
        rotation = torch.stack(
            (
                torch.stack((torch.cos(angle), -torch.sin(angle), angle.new_tensor(0.0))),
                torch.stack((torch.sin(angle), torch.cos(angle), angle.new_tensor(0.0))),
                angle.new_tensor([0.0, 0.0, 1.0]),
            )
        )
        tracked = torch.randn(6, generator=generator, dtype=torch.float64)
        canonical = torch.randn(7, generator=generator, dtype=torch.float64) * 0.05

        raw = encode_complete_action(
            canonical, candidate, ee_rotation=rotation, tracked_target=tracked
        )
        decoded, next_target = decode_complete_action(
            raw, candidate, ee_rotation=rotation, tracked_target=tracked
        )

        torch.testing.assert_close(decoded, canonical, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(next_target, tracked + canonical[:6])
