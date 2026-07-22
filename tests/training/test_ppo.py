from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from actionshift.policies.actor_critic import ActorCritic
from actionshift.training.checkpoint import load_checkpoint, save_checkpoint
from actionshift.training.ppo import PPOBatch, ppo_update


def test_ppo_update_is_finite_and_changes_parameters() -> None:
    torch.manual_seed(7)
    model = ActorCritic(4, 2, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    observations = torch.randn(32, 4)
    with torch.no_grad():
        actions, old_log_probabilities, _entropy, values = model.sample(observations)
    batch = PPOBatch(
        observations=observations,
        actions=actions,
        old_log_probabilities=old_log_probabilities,
        returns=values + 0.5,
        advantages=torch.ones(32),
        context=None,
    )
    before = [parameter.detach().clone() for parameter in model.parameters()]

    metrics = ppo_update(model, optimizer, batch, epochs=2, minibatch_size=8)

    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.parameters(), strict=True)
    )


def test_checkpoint_is_atomic_hash_checked_and_resumable() -> None:
    model = ActorCritic(4, 2, hidden_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.pt"
        digest = save_checkpoint(
            path, model=model, optimizer=optimizer, step=123, metadata={"seed": 7}
        )
        loaded = load_checkpoint(path, model=model, optimizer=optimizer, expected_sha256=digest)

        assert not (path.parent / f".{path.name}.tmp").exists()

    assert loaded.step == 123
    assert loaded.metadata == {"seed": 7}
