from __future__ import annotations

import torch

from actionshift.envs.safety import SafetyBounds


def test_safety_mask_rejects_action_velocity_workspace_gripper_and_collision() -> None:
    bounds = SafetyBounds(
        action_lower=torch.full((7,), -0.5),
        action_upper=torch.full((7,), 0.5),
        workspace_lower=torch.full((3,), -1.0),
        workspace_upper=torch.full((3,), 1.0),
        max_translation_norm=0.3,
        max_gripper_magnitude=0.8,
        max_collision_risk=0.2,
    )
    candidates = torch.zeros(1, 5, 7)
    candidates[0, 1, 0] = 0.6
    candidates[0, 2, :3] = torch.tensor([0.25, 0.25, 0.0])
    candidates[0, 3, -1] = 0.9
    predicted_displacement = candidates[..., :3].clone()
    predicted_displacement[0, 4, 0] = 0.2
    collision = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.3]])

    valid = bounds.mask(
        candidates,
        current_position=torch.tensor([[0.9, 0.0, 0.0]]),
        predicted_displacement=predicted_displacement,
        collision_risk=collision,
    )

    torch.testing.assert_close(valid, torch.tensor([[True, False, False, False, False]]))


def test_safety_mask_fails_closed_on_nonfinite_candidate() -> None:
    bounds = SafetyBounds.symmetric(action_dim=7)
    candidates = torch.zeros(1, 1, 7)
    candidates[0, 0, 0] = torch.nan

    valid = bounds.mask(
        candidates,
        current_position=torch.zeros(1, 3),
        predicted_displacement=torch.zeros(1, 1, 3),
        collision_risk=torch.zeros(1, 1),
    )

    assert not valid.item()
