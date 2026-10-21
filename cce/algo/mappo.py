"""MAPPO with communication bottleneck and budget control."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cce.envs.grid_world import VectorGridWorld
from cce.algo.networks import Actor, Critic
from cce.algo.buffer import RolloutBuffer
from cce.utils.logging import Logger


class ValueNormalizer:
    """Running mean/var normalizer for value targets (PopArt-lite)."""

    def __init__(self, shape: tuple[int, ...], device: torch.device):
        self.shape = shape
        self.device = device
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = 1e-5

    def update(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            batch_mean = x.mean()
            batch_var = x.var()
            batch_count = x.numel()

            delta = batch_mean - self.mean
            tot_count = self.count + batch_count
            new_mean = self.mean + delta * batch_count / tot_count
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
            new_var = M2 / tot_count

            self.mean = new_mean
            self.var = new_var
            self.count = tot_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (torch.sqrt(self.var) + 1e-8)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * (torch.sqrt(self.var) + 1e-8) + self.mean


class MAPPO:
    """Multi-Agent PPO with communication bottleneck."""

    def __init__(
        self,
        env: VectorGridWorld,
        actor_hidden: int = 64,
        critic_hidden: int = 256,
        backbone: str = "mlp",
        gru_hidden: int = 128,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_clip: float = 10.0,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        num_epochs: int = 10,
        batch_size: int = 3200,
        rollout_length: int | None = None,
        # Communication
        n_codes: int = 16,
        code_dim: int = 8,
        n_slots: int = 4,
        target_beta: float = 0.25,
        budget_mode: str = "policy",
        # Value normalisation
        use_value_norm: bool = True,
        # Critic type
        critic_type: str = "cnn",
    ):
        self.env = env
        self.B = env.B
        self.N = env.N
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor
        self.actor = Actor(
            obs_spec=env.obs_spec,
            n_actions=env.num_actions,
            hidden_dim=actor_hidden,
            backbone=backbone,
            gru_hidden=gru_hidden,
            n_codes=n_codes,
            code_dim=code_dim,
            n_slots=n_slots,
            budget_mode=budget_mode,
            target_beta=target_beta,
        ).to(self.device)

        # Critic
        self.critic = Critic(
            grid_size=env.G,
            num_agents=env.N,
            hidden_dim=critic_hidden,
            critic_type=critic_type,
        ).to(self.device)

        # Optimisers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr_actor, eps=1e-5)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)

        # Value normaliser
        self.value_norm = ValueNormalizer((), self.device) if use_value_norm else None

        # Hyperparams
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_clip = value_clip
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.backbone = backbone
        self.gru_hidden_dim = gru_hidden if backbone == "gru" else 0

        # Rollout length: default to full episode
        self.rollout_length = rollout_length or env.cfg.max_steps

        # State spec for centralised critic
        self.state_spec = {
            "state_map": (3, env.G, env.G),
            "agent_positions": (env.N, 2),
        }

        # Buffer
        self.buffer = RolloutBuffer(
            num_envs=self.B,
            num_agents=self.N,
            rollout_length=self.rollout_length,
            obs_spec=env.obs_spec,
            state_spec=self.state_spec,
            gru_hidden_dim=self.gru_hidden_dim,
            gamma=gamma,
            gae_lambda=gae_lambda,
            device=self.device,
        )

        # Metrics
        self.total_steps = 0
        self.num_updates = 0

    @torch.no_grad()
    def select_actions(
        self,
        obs: dict[str, np.ndarray],
        rnn_hidden: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Select actions for a batch of observations."""
        # Move obs to device
        obs_t = {k: torch.tensor(v, device=self.device) for k, v in obs.items()}
        hidden_t = None
        if rnn_hidden is not None:
            hidden_t = torch.tensor(rnn_hidden, device=self.device)

        # Forward actor
        dist, msg_info, hidden_new = self.actor(obs_t, hidden_t)
        actions_move, actions_send = dist.sample()
        log_probs = dist.log_prob(actions_move, actions_send)

        # Forward critic
        state = self.env.get_global_state()
        state_t = {k: torch.tensor(v, device=self.device) for k, v in state.items()}
        values = self.critic(state_t).squeeze(-1)  # (B,)

        # Broadcast values to (B, N) since shared reward
        values = values.unsqueeze(1).expand(self.B, self.N)

        # Convert to numpy
        actions_move_np = actions_move.cpu().numpy()
        actions_send_np = actions_send.cpu().numpy()
        log_probs_np = log_probs.cpu().numpy()
        values_np = values.cpu().numpy()

        info = {
            "message_indices": msg_info["message_indices"].cpu().numpy(),
            "message_quantised": msg_info["message_quantised"].cpu().numpy(),
            "vq_loss": msg_info["vq_loss"].item(),
            "perplexity": msg_info["perplexity"].item(),
        }

        if hidden_new is not None and hidden_new.numel() > 0:
            info["rnn_hidden"] = hidden_new.cpu().numpy()

        return actions_move_np, actions_send_np, log_probs_np, values_np, info

    def collect_rollout(self) -> dict[str, float]:
        """Collect one rollout of data from the environment."""
        obs = self.env.reset()
        rnn_hidden = self.actor.init_hidden(self.B * self.N, self.device).cpu().numpy()
        if self.backbone == "gru":
            rnn_hidden = rnn_hidden.reshape(self.B, self.N, self.gru_hidden_dim)
        else:
            rnn_hidden = None

        episode_rewards = []
        episode_coverages = []
        episode_t95 = []
        episode_comm_costs = []

        # Reset budget controller per episode
        if hasattr(self.actor, 'budget_controller'):
            self.actor.budget_controller.reset_episode()

        for t in range(self.rollout_length):
            # Select actions
            actions_move, actions_send, log_probs, values, info = self.select_actions(obs, rnn_hidden)

            # Compute comm cost
            n_send = actions_send.sum(axis=1)
            comm_cost = n_send / self.N
            
            if hasattr(self.actor, 'budget_controller'):
                self.actor.budget_controller.on_step(comm_cost.mean())

            # Build message embeddings for the env: (B, N, code_dim)
            message_embeddings = info["message_quantised"].mean(axis=-2)

            # Step env
            obs, rewards, dones, env_infos = self.env.step(
                actions_move, actions_send.astype(bool), message_embeddings
            )

            # Get global state for critic storage
            current_state = self.env.get_global_state()

            # Insert into buffer
            self.buffer.insert(
                obs=obs,
                actions_move=actions_move,
                actions_send=actions_send,
                log_probs=log_probs,
                rewards=rewards,     
                values=values,       
                dones=dones,
                comm_cost=comm_cost,
                state=current_state,
                rnn_hidden=info.get("rnn_hidden"),
            )

            # Update hidden state
            if info.get("rnn_hidden") is not None:
                rnn_hidden = info["rnn_hidden"]

            # Episode-level stats
            if dones.any():
                for b in range(self.B):
                    if dones[b]:
                        episode_rewards.append(float(rewards[b].sum()))
                        episode_coverages.append(float(env_infos["coverage"][b]))
                        episode_t95.append(int(env_infos["t95"][b]))
                        episode_comm_costs.append(float(comm_cost[b]))

        # Bootstrap value at end of rollout
        with torch.no_grad():
            obs_t = {k: torch.tensor(v, device=self.device) for k, v in obs.items()}
            hidden_t = None
            if rnn_hidden is not None:
                hidden_t = torch.tensor(rnn_hidden, device=self.device).reshape(-1, self.gru_hidden_dim)
            _, _, hidden_final = self.actor(obs_t, hidden_t)
            state = self.env.get_global_state()
            state_t = {k: torch.tensor(v, device=self.device) for k, v in state.items()}
            next_value = self.critic(state_t).squeeze(-1)
            next_value = next_value.unsqueeze(1).expand(self.B, self.N)

        self.buffer.compute_returns(next_value.cpu().numpy(), dones)

        # Update budget controller
        budget_stats = {"actual_beta": 0.0, "lambda": 0.0}
        if hasattr(self.actor, 'budget_controller'):
            budget_stats = self.actor.budget_controller.on_episode_end()

        # Aggregate metrics
        stats = {
            "episode_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "coverage": np.mean(episode_coverages) if episode_coverages else 0.0,
            "t95": np.mean([x for x in episode_t95 if x >= 0]) if any(x >= 0 for x in episode_t95) else -1,
            "comm_cost": np.mean(episode_comm_costs) if episode_comm_costs else 0.0,
            "actual_beta": budget_stats.get("actual_beta", 0.0),
            "lambda": budget_stats.get("lambda", 0.0),
            "vq_loss": info["vq_loss"],
            "perplexity": info["perplexity"],
        }

        return stats

    def update(self) -> dict[str, float]:
        """Run PPO update on collected rollout."""
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_vq_loss = 0.0
        n_batches = 0

        for batch in self.buffer.get_batches(self.batch_size, self.num_epochs):
            # Reconstruct obs dict
            obs_t = {}
            for k, v in batch.items():
                if k.startswith("obs_"):
                    obs_key = k[4:]  # remove "obs_" prefix
                    obs_t[obs_key] = v

            # Forward actor
            hidden = batch.get("rnn_hidden")
            dist, msg_info, _ = self.actor(obs_t, hidden)
            log_probs_new = dist.log_prob(batch["actions_move"], batch["actions_send"])
            entropy = dist.entropy().mean()

            # Forward critic on stored states from the batch
            if "state" in batch:
                state_t = batch["state"]  
                values = self.critic(state_t).squeeze(-1)  
            else:
                values = batch["values_old"]

            # Normalise advantages
            advantages = batch["advantages"]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy loss (clipped)
            ratio = torch.exp(log_probs_new - batch["log_probs_old"])
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss (clipped value)
            returns_norm = self.value_norm.normalize(batch["returns"]) if self.value_norm else batch["returns"]
            values_norm = self.value_norm.normalize(values) if self.value_norm else values
            values_old_norm = self.value_norm.normalize(batch["values_old"]) if self.value_norm else batch["values_old"]

            values_clipped = values_old_norm + torch.clamp(
                values_norm - values_old_norm, -self.value_clip, self.value_clip
            )
            critic_loss1 = F.mse_loss(values_norm, returns_norm, reduction="none")
            critic_loss2 = F.mse_loss(values_clipped, returns_norm, reduction="none")
            critic_loss = torch.max(critic_loss1, critic_loss2).mean()

            # VQ loss
            vq_loss = msg_info["vq_loss"]

            # Total loss
            loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy + vq_loss

            # Optimise
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.actor_optim.step()
            self.critic_optim.step()

            # Update value normaliser
            if self.value_norm:
                self.value_norm.update(batch["returns"])

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()
            total_vq_loss += vq_loss.item()
            n_batches += 1

        self.num_updates += 1
        return {
            "actor_loss": total_actor_loss / max(n_batches, 1),
            "critic_loss": total_critic_loss / max(n_batches, 1),
            "entropy": total_entropy / max(n_batches, 1),
            "vq_loss": total_vq_loss / max(n_batches, 1),
        }

    def train_step(self, logger: Logger | None = None) -> dict[str, float]:
        """Collect rollout + update. Returns combined stats."""
        
        # ================= BOOTCAMP / CURRICULUM LOGIC =================
        self._update_idx = getattr(self, '_update_idx', 0) + 1
        warmup = self._update_idx <= 120  # BOOTCAMP horizon

        # 1. Make talking free and incentivized during warmup
        if hasattr(self.env, 'comm_cost_scale'):
            self.env.comm_cost_scale = 0.0 if warmup else 1.0
            self.env.send_bonus = 0.5 if warmup else 0.0

        # 2. Pin the budget controller (lambda) during warmup so it doesn't prematurely punish talking
        _bc = getattr(self.actor, 'budget_controller', None)
        if _bc is None:
            _bc = getattr(self, 'budget_controller', None)
            
        if _bc is not None and hasattr(_bc, 'lam'):
            if warmup:
                _bc.lam.fill_(0.0)
                if hasattr(_bc, 'integral'): 
                    _bc.integral.fill_(0.0)
                if hasattr(_bc, 'prev_error'): 
                    _bc.prev_error.fill_(0.0)
            else:
                # Tame the PID controller once the real constraints kick in
                _bc.pid_kp = 0.05
                _bc.pid_ki = 0.005
                _bc.pid_kd = 0.01
        # ===============================================================

        rollout_stats = self.collect_rollout()
        update_stats = self.update()

        self.total_steps += self.rollout_length * self.B

        stats = {**rollout_stats, **update_stats, "total_steps": self.total_steps}

        if logger is not None:
            logger.log(stats, step=self.total_steps)

        return stats