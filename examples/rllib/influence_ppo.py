import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque, defaultdict
from typing import List, Dict, Tuple, Optional
import wandb


def get_relative_positions(obs):
    """Extract relative positions from structured observations."""
    agent_pos = obs.get("WORLD.AVATAR.POSITION", np.zeros(2))  # shape: [2]
    entity_positions = obs.get(
        "WORLD.ENTITY_POSITIONS", np.zeros((0, 2))
    )  # shape: [N, 2]

    if len(entity_positions) == 0:
        return np.zeros((1, 2))  # Return dummy if no entities

    # Compute relative positions
    rel_positions = entity_positions - agent_pos  # shape: [N, 2]
    return rel_positions


def extract_structured_features(obs, agent_idx=None, all_agent_positions=None):
    """
    Extract structured features from ANY MeltingPot substrate automatically.
    Now includes relative positions of other agents.
    Always returns the same size vector regardless of substrate.

    Args:
        obs: The observation dictionary/array for this agent
        agent_idx: Index of this agent (for computing relative positions)
        all_agent_positions: List of all agent positions to compute relative positions
    """
    # Fixed feature vector size for all substrates
    BASE_SCALAR_SIZE = 20
    MAX_OTHER_AGENTS = 4  # Maximum number of other agents to consider
    FEATURE_SIZE = BASE_SCALAR_SIZE + MAX_OTHER_AGENTS * 2  # +2 for (x,y) per agent

    features = [0.0] * FEATURE_SIZE

    if isinstance(obs, dict):
        feature_idx = 0

        # List of all possible scalar observations across all substrates
        possible_scalars = [
            "COLLECTIVE_REWARD",
            "MISMATCHED_COIN_COLLECTED_BY_PARTNER",
            "READY_TO_SHOOT",
            "STAMINA",
            "HUNGER",
            "NUM_OTHERS_WHO_CLEANED_THIS_STEP",
        ]

        # Extract all available scalar features
        for obs_name in possible_scalars:
            if obs_name in obs and feature_idx < BASE_SCALAR_SIZE:
                val = obs[obs_name]
                if isinstance(val, (int, float, np.integer, np.floating)):
                    features[feature_idx] = float(val)
                    feature_idx += 1
                elif hasattr(val, "item"):  # Handle numpy scalars
                    features[feature_idx] = float(val.item())
                    feature_idx += 1

        # Add relative positions of other agents if available
        if all_agent_positions is not None and agent_idx is not None:
            my_pos = obs.get("WORLD.AVATAR.POSITION", np.zeros(2))
            rel_positions = []

            # Compute relative positions to other agents
            for i, other_pos in enumerate(all_agent_positions):
                if i == agent_idx:  # Skip self
                    continue
                rel = np.array(other_pos) - np.array(my_pos)
                rel_positions.extend(rel.tolist())

            # Pad or truncate to exact size
            rel_positions = rel_positions[:MAX_OTHER_AGENTS * 2]
            rel_positions += [0.0] * (MAX_OTHER_AGENTS * 2 - len(rel_positions))

            # Add relative positions to feature vector
            features[BASE_SCALAR_SIZE:] = rel_positions

    elif isinstance(obs, np.ndarray):
        # Handle array observations - use first BASE_SCALAR_SIZE elements
        obs_size = min(len(obs), BASE_SCALAR_SIZE)
        features[:obs_size] = obs[:obs_size].tolist()

    result = np.array(features, dtype=np.float32)

    # Debug info for dimension mismatches
    if len(result) != FEATURE_SIZE:
        print(f"[DEBUG] Feature size mismatch in extract_structured_features:")
        print(f"[DEBUG] Expected: {FEATURE_SIZE}, Got: {len(result)}")
        print(f"[DEBUG] Observation type: {type(obs)}")
        if isinstance(obs, dict):
            print(f"[DEBUG] Dict keys: {list(obs.keys())}")
        result = result[:FEATURE_SIZE] if len(result) > FEATURE_SIZE else np.pad(result, (0, FEATURE_SIZE - len(result)))

    return result


class StructuredActorCritic(nn.Module):
    """
    Actor-Critic network that processes structured observations directly.
    No CNN layers - only fully connected networks.
    Updated to handle relative positions in the input features.
    """

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # No CNN layers - direct processing of structured features including relative positions
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, structured_features: torch.Tensor):
        """
        Forward pass with structured features including relative positions.

        Args:
            structured_features: Tensor of shape [batch_size, input_dim]
                                Now includes relative positions of other agents
        """
        # Process structured features directly (including relative positions)
        features = self.feature_net(structured_features)

        # Get policy logits and value
        action_logits = self.actor(features)
        value = self.critic(features)

        return action_logits, value


class SocialInfluencePPO:
    """
    PPO algorithm with social influence rewards using structured observations.
    Enhanced with relative position awareness for better spatial coordination.
    No image/CNN processing - works with relative positions and structured data.
    """

    def __init__(
        self,
        env,
        num_agents: int,
        input_dim: int,  # Dimension of structured features (including relative positions)
        action_dim: int,
        influence_weight: float = 0.1,
        curriculum_steps: int = 0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        device: str = "cpu",
        use_wandb: bool = False,
    ):

        self.env = env
        self.num_agents = num_agents
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.device = device
        self.use_wandb = use_wandb

        # Influence parameters
        self.base_influence_weight = influence_weight
        self.curriculum_steps = curriculum_steps
        self.current_influence_weight = (
            0.0 if curriculum_steps > 0 else influence_weight
        )

        # PPO parameters
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        # Create networks for each agent
        self.agents = []
        self.optimizers = []
        for _ in range(num_agents):
            agent = StructuredActorCritic(input_dim, action_dim).to(device)
            self.agents.append(agent)
            self.optimizers.append(optim.Adam(agent.parameters(), lr=lr))

        # Storage for PPO
        self.memory = {
            "states": [],
            "actions": [],
            "logprobs": [],
            "rewards": [],
            "is_terminals": [],
            "values": [],
        }

        # Metrics
        def create_deque_1000():
            return deque(maxlen=1000)

        self.metrics = {
            "collective_rewards": deque(maxlen=1000),
            "individual_rewards": defaultdict(create_deque_1000),
            "influence_rewards": deque(maxlen=1000),
            "losses": deque(maxlen=1000),
            "episode_lengths": deque(maxlen=100),
            "relative_position_variance": deque(maxlen=1000),  # New metric for spatial spread
        }

        self.step_count = 0
        self.episode_count = 0

    def get_current_influence_weight(self):
        """Update influence weight based on curriculum."""
        if self.curriculum_steps > 0:
            progress = min(1.0, self.step_count / self.curriculum_steps)
            self.current_influence_weight = self.base_influence_weight * progress
        return self.current_influence_weight

    def compute_influence_reward(
        self, agent_idx: int, structured_states: List[torch.Tensor]
    ) -> float:
        """
        Compute social influence reward using policy divergence.
        Uses structured features including relative positions.

        The influence reward measures how much agent i changes other agents' policies.
        Higher KL divergence = more influence = higher reward.

        With relative positions, agents can now be more strategic about their influence
        based on spatial relationships.
        """
        if len(structured_states) <= 1:
            return 0.0

        total_influence = 0.0

        # For each other agent j
        for j in range(self.num_agents):
            if j == agent_idx:
                continue

            j_state = structured_states[j]

            # Get j's action probabilities WITH agent i present (includes i's position)
            with torch.no_grad():
                logits_with_i, _ = self.agents[j](j_state)
                probs_with_i = F.softmax(logits_with_i, dim=-1)

            # Create counterfactual: what would j do WITHOUT agent i's spatial influence?
            # For relative positions, we can either:
            # 1. Use uniform distribution (simple baseline)
            # 2. Use the agent's own policy from a previous state (more sophisticated)
            # Here we use approach 1 for simplicity
            uniform_probs = torch.ones_like(probs_with_i) / self.action_dim

            # Compute KL divergence: KL(P_with_i || P_without_i)
            # Add small epsilon to prevent log(0)
            eps = 1e-8
            probs_with_i_safe = probs_with_i + eps
            uniform_probs_safe = uniform_probs + eps

            kl_divergence = F.kl_div(
                uniform_probs_safe.log(), probs_with_i_safe, reduction="sum"
            ).item()

            total_influence += kl_divergence

        # Average influence across all other agents
        return total_influence / (self.num_agents - 1) if self.num_agents > 1 else 0.0

    def select_action(self, structured_features: torch.Tensor, agent_idx: int):
        """Select action using current policy (for structured features with relative positions)."""
        with torch.no_grad():
            logits, value = self.agents[agent_idx](structured_features)
            probs = F.softmax(logits, dim=-1)

            dist = Categorical(probs)
            action = dist.sample()

            return action.item(), dist.log_prob(action).item(), value.item()

    def train_episode(self):
        """
        Run one episode and collect data for PPO training.
        Now includes relative position computation and tracking.
        """
        # Get initial observations and convert to structured features
        raw_obs = self.env.reset()
        agent_positions = []
        structured_states = []

        # First pass: extract positions for all agents
        for agent_idx in range(self.num_agents):
            if isinstance(raw_obs, dict):
                agent_obs = raw_obs[f"player_{agent_idx}"]
            else:
                agent_obs = raw_obs[agent_idx]

            # Extract position information
            if isinstance(agent_obs, dict):
                pos = agent_obs.get("WORLD.AVATAR.POSITION", np.zeros(2))
            else:
                # Assuming position is first 2 elements if array observation
                pos = agent_obs[:2] if len(agent_obs) >= 2 else np.zeros(2)

            agent_positions.append(pos)

        # Second pass: create structured features including relative positions
        for agent_idx in range(self.num_agents):
            if isinstance(raw_obs, dict):
                agent_obs = raw_obs[f"player_{agent_idx}"]
            else:
                agent_obs = raw_obs[agent_idx]

            structured_features = extract_structured_features(
                agent_obs, agent_idx, agent_positions
            )
            structured_tensor = (
                torch.FloatTensor(structured_features).unsqueeze(0).to(self.device)
            )
            structured_states.append(structured_tensor)

        episode_reward = 0
        episode_length = 0
        position_variances = []  # Track spatial spread

        done = False
        while not done and episode_length < 1000:
            # Get actions for all agents
            actions = []
            log_probs = []
            values = []

            for i in range(self.num_agents):
                action, log_prob, value = self.select_action(structured_states[i], i)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)

            # Environment step
            step_return = self.env.step(actions)
            if len(step_return) == 4:
                next_raw_obs, rewards, done, info = step_return
            else:
                next_raw_obs, rewards, done, truncated, info = step_return
                done = done or truncated

            # Compute influence rewards
            influence_rewards = []
            for i in range(self.num_agents):
                influence = self.compute_influence_reward(i, structured_states)
                influence_rewards.append(influence)

            # Combine environment and influence rewards
            current_influence_weight = self.get_current_influence_weight()
            modified_rewards = []
            for i in range(self.num_agents):
                total_reward = (
                    rewards[i] + current_influence_weight * influence_rewards[i]
                )
                modified_rewards.append(total_reward)

            # Store in PPO memory
            self.memory["states"].extend(structured_states)
            self.memory["actions"].extend(actions)
            self.memory["logprobs"].extend(log_probs)
            self.memory["rewards"].extend(modified_rewards)
            self.memory["values"].extend(values)
            self.memory["is_terminals"].extend([done] * self.num_agents)

            # Update states for next iteration with new relative positions
            agent_positions = []

            # First pass: extract new positions
            for agent_idx in range(self.num_agents):
                if isinstance(next_raw_obs, dict):
                    agent_obs = next_raw_obs[f"player_{agent_idx}"]
                else:
                    agent_obs = next_raw_obs[agent_idx]

                if isinstance(agent_obs, dict):
                    pos = agent_obs.get("WORLD.AVATAR.POSITION", np.zeros(2))
                else:
                    pos = agent_obs[:2] if len(agent_obs) >= 2 else np.zeros(2)

                agent_positions.append(pos)

            # Compute position variance for metrics
            if len(agent_positions) > 1:
                positions_array = np.array(agent_positions)
                position_variance = np.var(positions_array)
                position_variances.append(position_variance)

            # Second pass: create new structured states with relative positions
            structured_states = []
            for agent_idx in range(self.num_agents):
                if isinstance(next_raw_obs, dict):
                    agent_obs = next_raw_obs[f"player_{agent_idx}"]
                else:
                    agent_obs = next_raw_obs[agent_idx]

                structured_features = extract_structured_features(
                    agent_obs, agent_idx, agent_positions
                )
                structured_tensor = (
                    torch.FloatTensor(structured_features).unsqueeze(0).to(self.device)
                )
                structured_states.append(structured_tensor)

            episode_reward += info.get("collective_reward", sum(rewards))
            episode_length += 1

        # Train PPO after episode
        if len(self.memory["states"]) > 0:
            self._update_ppo()

        # Update metrics
        self.metrics["collective_rewards"].append(episode_reward)
        self.metrics["episode_lengths"].append(episode_length)
        if position_variances:
            avg_position_variance = np.mean(position_variances)
            self.metrics["relative_position_variance"].append(avg_position_variance)

        self.step_count += episode_length
        self.episode_count += 1

        # Log to wandb if enabled
        if self.use_wandb:
            log_dict = {
                "episode_reward": episode_reward,
                "episode_length": episode_length,
                "influence_weight": current_influence_weight,
                "episode": self.episode_count,
                "avg_influence_reward": np.mean(influence_rewards) if influence_rewards else 0,
            }

            if position_variances:
                log_dict["avg_position_variance"] = avg_position_variance

            wandb.log(log_dict)

        return episode_reward

    def _update_ppo(self):
        """Update PPO networks using collected experiences with relative position features."""
        # Convert lists to tensors
        states = torch.cat(self.memory["states"]).detach()
        actions = torch.LongTensor(self.memory["actions"]).to(self.device)
        old_logprobs = torch.FloatTensor(self.memory["logprobs"]).to(self.device)
        rewards = torch.FloatTensor(self.memory["rewards"]).to(self.device)
        values = torch.FloatTensor(self.memory["values"]).to(self.device)
        is_terminals = torch.BoolTensor(self.memory["is_terminals"]).to(self.device)

        # Compute discounted rewards
        discounted_rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            discounted_rewards.insert(0, discounted_reward)

        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)

        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + 1e-8
        )

        # Calculate advantages
        advantages = discounted_rewards - values.detach()

        # PPO update for each agent separately
        total_loss = 0
        for epoch in range(self.k_epochs):
            for agent_idx in range(self.num_agents):
                # Get data for this agent (every num_agents-th element starting from agent_idx)
                agent_indices = list(range(agent_idx, len(states), self.num_agents))
                if not agent_indices:
                    continue

                agent_states = states[agent_indices]
                agent_actions = actions[agent_indices]
                agent_old_logprobs = old_logprobs[agent_indices]
                agent_advantages = advantages[agent_indices]
                agent_discounted_rewards = discounted_rewards[agent_indices]

                # Evaluate actions with current policy
                logits, state_values = self.agents[agent_idx](agent_states)

                dist = Categorical(F.softmax(logits, dim=-1))
                new_logprobs = dist.log_prob(agent_actions)
                entropy = dist.entropy()

                # Calculate PPO loss
                ratios = torch.exp(new_logprobs - agent_old_logprobs.detach())

                surr1 = ratios * agent_advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * agent_advantages
                )

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(state_values.squeeze(), agent_discounted_rewards)
                entropy_loss = -entropy.mean()

                total_loss_agent = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
                total_loss += total_loss_agent.item()

                # Update network
                self.optimizers[agent_idx].zero_grad()
                total_loss_agent.backward()
                torch.nn.utils.clip_grad_norm_(self.agents[agent_idx].parameters(), 0.5)
                self.optimizers[agent_idx].step()

        self.metrics["losses"].append(total_loss / (self.k_epochs * self.num_agents))

        # Clear memory
        self.memory = {
            "states": [],
            "actions": [],
            "logprobs": [],
            "rewards": [],
            "is_terminals": [],
            "values": [],
        }

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "agent_states": [agent.state_dict() for agent in self.agents],
            "optimizer_states": [opt.state_dict() for opt in self.optimizers],
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath)

        self.step_count = checkpoint["step_count"]
        self.episode_count = checkpoint["episode_count"]

        for i, (agent, opt) in enumerate(zip(self.agents, self.optimizers)):
            agent.load_state_dict(checkpoint["agent_states"][i])
            opt.load_state_dict(checkpoint["optimizer_states"][i])


# Example usage
if __name__ == "__main__":
    # Initialize wandb
    wandb.init(project="social-influence-ppo-with-positions")

    # Example configuration - note the increased input_dim to account for relative positions
    num_agents = 5
    base_features = 20  # Base scalar features
    max_other_agents = 4  # Maximum other agents to consider
    input_dim = base_features + max_other_agents * 2  # +2 for (x,y) per other agent
    action_dim = 9  # Number of possible actions

    print(f"Using input dimension: {input_dim}")
    print(f"This includes {base_features} base features + {max_other_agents * 2} relative position features")

    # Create trainer (you'll need to provide your actual environment)
    trainer = SocialInfluencePPO(
        env=None,  # Your environment here
        num_agents=num_agents,
        input_dim=input_dim,
        action_dim=action_dim,
        influence_weight=0.1,
        use_wandb=True,
    )

    # Training loop
    for episode in range(1000):
        reward = trainer.train_episode()

        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {reward}")

            # Print some metrics about spatial coordination
            if len(trainer.metrics["relative_position_variance"]) > 0:
                avg_variance = np.mean(list(trainer.metrics["relative_position_variance"])[-100:])
                print(f"Average position variance (last 100 episodes): {avg_variance:.3f}")

            trainer.save_checkpoint(f"checkpoint_with_positions_episode_{episode}.pt")
