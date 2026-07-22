"""Faithful DCAC-style constant-delay wrapper for Gymnasium MuJoCo envs.

This reproduces the delay-injection setup of Bouteiller et al. 2020,
"Reinforcement Learning with Random Delays" (arXiv:2010.02966), whose released
Gym wrapper (github.com/rmst/rlrd) turns any turn-based Gym env into a
Random-Delay MDP (RDMDP) parameterised by an *observation delay* omega and an
*action delay* alpha (in timesteps). Their headline constant-delay experiment
(Figure 6) uses ``omega=2, alpha=3`` -> a constant TOTAL delay of five timesteps.

We implement the classical augmented-state reduction (Katsikopoulos & Engelbrecht
2003; Walsh et al. 2009): the delayed MDP is made Markov again by augmenting the
(delayed) observation with the buffer of the last ``K = omega + alpha`` actions the
agent has emitted. This is exactly the augmentation ActionShift's
"delay-aware augmented-state PPO (local)" uses (obs + last-K actions); here we
port it onto THEIR benchmark (MuJoCo locomotion + their omega/alpha delay spec)
so our numbers can be read against their published curves.

Semantics (per-env, applied before vectorisation):
  * action delay alpha: the action the underlying env executes at agent-step t is
    the action the agent selected alpha steps earlier (a FIFO action pipeline of
    length alpha, zero-initialised at reset). Matches rlrd's action-delay channel.
  * observation delay omega: the observation handed back to the agent is the true
    state from omega steps ago (a FIFO obs pipeline of length omega).
  * augmentation: returned observation = concat(delayed_obs, last K agent actions),
    K = omega + alpha, so the tuple is a sufficient statistic for the constant-delay
    MDP. Rewards are the true per-transition rewards (episode RETURN is invariant to
    reward-delay, and associating reward with the current augmented state is the
    standard constant-delay treatment).

Only the constant-delay case is implemented (their headline Fig-6 config); random
WiFi delays (Fig 7) are out of scope for this harness and documented as such in the
report.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np


class ConstantDelayWrapper(gym.Wrapper):
    """Constant observation+action delay with augmented-state observations."""

    def __init__(
        self, env: gym.Env, obs_delay: int = 2, act_delay: int = 3, augment: bool = True
    ):
        super().__init__(env)
        assert obs_delay >= 0 and act_delay >= 0
        self.obs_delay = int(obs_delay)
        self.act_delay = int(act_delay)
        # augmentation history length: K = omega + alpha when augmenting, else 0.
        # augment=False is the NAIVE control (delay still injected, but the agent
        # gets NO last-K-action history -> the delayed MDP stays non-Markov, which
        # DCAC reports drives naive SAC to "near-random" returns).
        self.augment = bool(augment)
        self.k = (self.obs_delay + self.act_delay) if self.augment else 0

        base_obs = env.observation_space
        assert isinstance(base_obs, gym.spaces.Box) and len(base_obs.shape) == 1
        self.act_dim = int(np.prod(env.action_space.shape))
        aug_dim = int(base_obs.shape[0]) + self.k * self.act_dim
        high = np.full(aug_dim, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        self._act_pipeline: deque[np.ndarray] = deque()
        self._obs_pipeline: deque[np.ndarray] = deque()
        self._act_history: deque[np.ndarray] = deque()

    def _augment(self, delayed_obs: np.ndarray) -> np.ndarray:
        hist = np.concatenate(list(self._act_history)) if self.k else np.zeros(0, np.float32)
        return np.concatenate([delayed_obs.astype(np.float32), hist.astype(np.float32)])

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        zero_a = np.zeros(self.act_dim, dtype=np.float32)
        # action pipeline: alpha zero-actions in flight
        self._act_pipeline = deque(
            [zero_a.copy() for _ in range(self.act_delay)], maxlen=max(self.act_delay, 1)
        )
        # obs pipeline: omega copies of the initial obs already "in flight"
        self._obs_pipeline = deque(
            [obs.copy() for _ in range(self.obs_delay)], maxlen=max(self.obs_delay, 1)
        )
        # action history for augmentation: K zero actions
        self._act_history = deque([zero_a.copy() for _ in range(self.k)], maxlen=max(self.k, 1))
        delayed_obs = obs.copy() if self.obs_delay == 0 else self._obs_pipeline[0]
        return self._augment(delayed_obs), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        # record what the agent emitted (for the augmentation buffer)
        if self.k:
            self._act_history.append(action.copy())
        # action delay: env executes the alpha-steps-old action
        if self.act_delay == 0:
            executed = action
        else:
            executed = self._act_pipeline.popleft()
            self._act_pipeline.append(action.copy())
        obs, reward, terminated, truncated, info = self.env.step(executed)
        # observation delay: agent sees the omega-steps-old obs
        if self.obs_delay == 0:
            delayed_obs = obs
        else:
            delayed_obs = self._obs_pipeline.popleft()
            self._obs_pipeline.append(obs.copy())
        return self._augment(delayed_obs), float(reward), bool(terminated), bool(truncated), info
