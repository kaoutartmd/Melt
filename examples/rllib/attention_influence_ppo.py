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


def extract_items_positions(obs):
    """Extract item/resource positions from environment observations.

    This function extracts positions of collectible items (apples, berries, etc.)
    from MeltingPot environment observations. Works across different substrates.
    """
    agent_pos = obs.get("WORLD.AVATAR.POSITION", np.zeros(2))

    # Look for various item types that might exist in different MeltingPot substrates
    item_keys = [
        "WORLD.APPLES",  # For harvest/gathering games
        "WORLD.BERRIES",  # For some foraging games
        "WORLD.RESOURCES",  # Generic resources
        "WORLD.COLLECTIBLES",  # Generic collectibles
        "WORLD.ITEMS",  # Generic items
        "WORLD.FOOD",  # Food items
    ]

    all_item_positions = []

    for key in item_keys:
        if key in obs:
            items_data = obs[key]
            if isinstance(items_data, np.ndarray) and len(items_data.shape) == 2:
                # Assume format is [N, 2] for positions
                all_item_positions.extend(items_data)
            elif isinstance(items_data, list):
                # Handle list of positions
                for item in items_data:
                    if hasattr(item, "position") or isinstance(
                        item, (list, tuple, np.ndarray)
                    ):
                        pos = item.position if hasattr(item, "position") else item
                        if len(pos) >= 2:
                            all_item_positions.append(pos[:2]) #take the first two coordinates(x,y)

    if not all_item_positions:
        return np.zeros((0, 2))

    # Convert to numpy array and compute relative positions
    item_positions = np.array(all_item_positions)
    relative_items = item_positions - agent_pos

    return relative_items


def extract_structured_features(obs, agent_idx=None, all_agent_positions=None):
    """Handle both dictionary and array observations with robust error handling."""
    BASE_SCALAR_SIZE = 20
    MAX_OTHER_AGENTS = 4
    FEATURE_SIZE = BASE_SCALAR_SIZE + MAX_OTHER_AGENTS * 2
    features = [0.0] * FEATURE_SIZE

    if isinstance(obs, dict):
        # Handle dictionary observations (original code)
        feature_idx = 0
        possible_scalars = [
            "COLLECTIVE_REWARD",
            "MISMATCHED_COIN_COLLECTED_BY_PARTNER",
            "READY_TO_SHOOT",
            "STAMINA",
            "HUNGER",
            "NUM_OTHERS_WHO_CLEANED_THIS_STEP",
        ]

        for obs_name in possible_scalars:
            if obs_name in obs and feature_idx < BASE_SCALAR_SIZE:
                val = obs[obs_name]
                if isinstance(val, (int, float, np.integer, np.floating)):
                    features[feature_idx] = float(val)
                    feature_idx += 1
                elif hasattr(val, "item"):
                    features[feature_idx] = float(val.item())
                    feature_idx += 1

        # Handle position data if available
        if all_agent_positions is not None and agent_idx is not None:
            my_pos = obs.get("WORLD.AVATAR.POSITION", np.zeros(2))
            rel_positions = []
            for i, other_pos in enumerate(all_agent_positions):
                if i == agent_idx:
                    continue
                rel = np.array(other_pos) - np.array(my_pos)
                rel_positions.extend(rel.tolist())

            rel_positions = rel_positions[: MAX_OTHER_AGENTS * 2]
            rel_positions += [0.0] * (MAX_OTHER_AGENTS * 2 - len(rel_positions))
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
        result = (
            result[:FEATURE_SIZE]
            if len(result) > FEATURE_SIZE
            else np.pad(result, (0, FEATURE_SIZE - len(result)))
        )

    return result


class AttentionEmbeddingModel(nn.Module):
    """
    New attention model that considers both neighbors (other agents) and items.
    FIXED VERSION: Handles empty tensors gracefully to prevent index errors.

    This model follows your specified approach:
    1. Embed neighbors and items separately
    2. Compute mean embeddings
    3. Use keys from neighbor embeddings
    4. Compute attention weights using fa(mean_item_emb, emb_neigh_i, key_i)
    5. Return weighted sum over neighbor embeddings
    """

    def __init__(self, position_dim=2, hidden_dim=128):
        super().__init__()
        self.neighbor_embed_net = nn.Linear(
            position_dim, hidden_dim
        )  # e_i for neighbors
        self.item_embed_net = nn.Linear(position_dim, hidden_dim)  # a_j for items
        self.key_net = nn.Linear(hidden_dim, hidden_dim)  # key(e_i)
        # attention: fa(mean_item_emb, emb_neigh_i, key_i)
        self.attention_net = nn.Linear(hidden_dim * 3, 1)

    def forward(
        self, relative_neighbors: torch.Tensor, relative_items: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            relative_neighbors: [N, 2] - relative positions of neighbors
            relative_items: [M, 2] - relative positions of items/resources
        Returns:
            context vector: [hidden_dim]
            mean_item_emb: [hidden_dim]
            attn_weights: [N] - attention weights for neighbors
        """
        hidden_dim = self.neighbor_embed_net.out_features
        device = relative_neighbors.device

        # Handle empty neighbors case
        if relative_neighbors.size(0) == 0:
            context = torch.zeros(hidden_dim, device=device)
            mean_item_emb = torch.zeros(hidden_dim, device=device)
            attn_weights = torch.empty(0, device=device)

            # If we have items but no neighbors, still compute mean item embedding
            if relative_items.size(0) > 0:
                item_embs = self.item_embed_net(relative_items)
                mean_item_emb = item_embs.mean(dim=0)

            return context, mean_item_emb, attn_weights

        # Handle empty items case
        if relative_items.size(0) == 0:
            # Use zero embeddings for items when none are present
            mean_item_emb = torch.zeros(hidden_dim, device=device)

            # Still compute neighbor embeddings and uniform attention
            neighbor_embs = self.neighbor_embed_net(relative_neighbors)  # [N, H]
            num_neighbors = neighbor_embs.size(0)

            # Uniform attention weights when no items to guide attention
            attn_weights = torch.ones(num_neighbors, device=device) / num_neighbors
            context = torch.mean(neighbor_embs, dim=0)  # Simple mean

            return context, mean_item_emb, attn_weights

        # Normal case: both neighbors and items present
        # Step 1: Embed neighbors and items
        neighbor_embs = self.neighbor_embed_net(relative_neighbors)  # [N, H]
        item_embs = self.item_embed_net(relative_items)  # [M, H]

        # Step 2: Mean embeddings
        mean_item_emb = item_embs.mean(dim=0)  # [H] - mean of all items

        # Step 3: Compute keys from neighbor embeddings
        keys = self.key_net(neighbor_embs)  # [N, H]

        # Step 4: Compute attention weights α_i = fa(mean_item, emb_i, key_i)
        repeated_mean_item = mean_item_emb.expand(neighbor_embs.size(0), -1)  # [N, H]
        attn_input = torch.cat(
            [repeated_mean_item, neighbor_embs, keys], dim=-1
        )  # [N, 3H]
        attn_logits = self.attention_net(attn_input).squeeze(-1)  # [N]
        attn_weights = F.softmax(attn_logits, dim=-1)  # [N]

        # Step 5: Weighted sum over neighbor embeddings (NO key in the final sum)
        context = torch.sum(attn_weights.unsqueeze(-1) * neighbor_embs, dim=0)  # [H]

        return context, mean_item_emb, attn_weights


class AttentionActorCritic(nn.Module):
    """
    Updated Actor-Critic network using the new AttentionEmbeddingModel.

    Now considers both neighbor agents AND items in the environment,
    making it more aware of the full game state.
    """

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        # Use new attention model that considers both neighbors and items
        self.attention = AttentionEmbeddingModel(position_dim=2, hidden_dim=32)

        # Feature network takes: original features + context + mean_item_embedding
        self.feature_net = nn.Sequential(
            nn.Linear(
                feature_dim + 32 + 32, hidden_dim
            ),  # +64 for context, +64 for mean_item_emb
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        my_features: torch.Tensor,
        neighbor_positions: torch.Tensor,
        item_positions: torch.Tensor,
    ):
        """
        Args:
            my_features: [feature_dim] - agent's own features
            neighbor_positions: [N, 2] - relative positions of other agents
            item_positions: [M, 2] - relative positions of items/resources
        """
        # Get attention context considering both neighbors and items
        context, mean_item_emb, attention_weights = self.attention(
            neighbor_positions, item_positions
        )

        # Combine agent's own features with social context and item awareness
        combined_features = torch.cat([my_features, context, mean_item_emb], dim=-1)
        features = self.feature_net(combined_features)

        action_logits = self.actor(features)
        value = self.critic(features)

        return action_logits, value, attention_weights


class AttentionInfluencePPO:
    def __init__(
        self,
        env,
        num_agents: int,
        feature_dim: int,
        action_dim: int,
        influence_weight: float = 0.01,
        curriculum_steps: int = 0,
        lr: float = 1e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        device: str = "cpu",
        use_wandb: bool = False,
    ):

        self.env = env
        self.num_agents = num_agents
        self.feature_dim = feature_dim
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

        # Create attention-based networks for each agent
        self.agents = []
        self.optimizers = []
        for _ in range(num_agents):
            agent = AttentionActorCritic(feature_dim, action_dim).to(device)
            self.agents.append(agent)
            self.optimizers.append(optim.Adam(agent.parameters(), lr=lr))

        # Initialize memory
        self.reset_memory()

        # Metrics
        def create_deque_1000():
            return deque(maxlen=1000)

        self.metrics = {
            "collective_rewards": deque(maxlen=1000),
            "individual_rewards": defaultdict(create_deque_1000),
            "influence_rewards": deque(maxlen=1000),
            "attention_entropy": deque(maxlen=1000),
            "losses": deque(maxlen=1000),
            "episode_lengths": deque(maxlen=100),
            "item_attention_correlation": deque(maxlen=1000),  # New metric
        }

        self.step_count = 0
        self.episode_count = 0

    def reset_memory(self):
        """Initialize or reset the memory buffers"""
        self.memory = {
            "states": [],
            "neighbor_positions": [],
            "item_positions": [],
            "actions": [],
            "logprobs": [],
            "rewards": [],
            "is_terminals": [],
            "values": [],
            "attention_weights": [],
        }

    def get_current_influence_weight(self):
        """Update influence weight based on curriculum."""
        if self.curriculum_steps > 0:
            progress = min(1.0, self.step_count / self.curriculum_steps)
            self.current_influence_weight = self.base_influence_weight * progress
        return self.current_influence_weight

    def compute_attention_influence_reward(
        self,
        agent_idx: int,
        structured_states: List[torch.Tensor],
        neighbor_positions_list: List[torch.Tensor],
        item_positions_list: List[torch.Tensor],
        neighbor_indices: List[int],
        attention_weights: torch.Tensor,
    ) -> float:
        """
        Updated influence computation considering item-aware attention.
        FIXED VERSION: Handles empty tensors and edge cases gracefully.

        The influence is now based on how the agent's presence affects
        other agents' policies when considering both social and item contexts.
        """
        if (
            not neighbor_indices
            or attention_weights is None
            or attention_weights.numel() == 0
        ):
            return 0.0

        total_influence = 0.0

        for idx, j in enumerate(neighbor_indices):
            try:
                j_features = structured_states[j]
                j_neighbor_positions = neighbor_positions_list[j]
                j_item_positions = item_positions_list[j]

                # Get j's action probabilities WITH agent i present
                with torch.no_grad():
                    logits_with_i, _, _ = self.agents[j](
                        j_features, j_neighbor_positions, j_item_positions
                    )
                    probs_with_i = F.softmax(logits_with_i, dim=-1)

                # Get j's action probabilities WITHOUT agent i present
                # Remove agent i from j's neighbor positions
                j_neighbors_without_i = []
                for k, neighbor_idx in enumerate(neighbor_indices):
                    if neighbor_idx != agent_idx and neighbor_idx != j:
                        # Find position of agent k relative to agent j
                        if k < len(neighbor_positions_list[j]):
                            k_pos_relative_to_j = neighbor_positions_list[j][k]
                            j_neighbors_without_i.append(k_pos_relative_to_j)

                # Handle empty neighbor list case
                if len(j_neighbors_without_i) > 0:
                    j_neighbors_without_i_tensor = torch.stack(j_neighbors_without_i)
                else:
                    j_neighbors_without_i_tensor = torch.zeros(
                        (0, 2), device=self.device
                    )

                with torch.no_grad():
                    logits_without_i, _, _ = self.agents[j](
                        j_features, j_neighbors_without_i_tensor, j_item_positions
                    )
                    probs_without_i = F.softmax(logits_without_i, dim=-1)

                # Compute KL divergence with numerical stability
                # Add small epsilon to prevent log(0)
                eps = 1e-8
                probs_with_i_safe = probs_with_i + eps
                probs_without_i_safe = probs_without_i + eps

                kl_divergence = F.kl_div(
                    probs_without_i_safe.log(), probs_with_i_safe, reduction="sum"
                ).item()

                # Ensure we have valid attention weight index
                if idx < len(attention_weights):
                    attention_weight = attention_weights[idx].item()
                    weighted_influence = attention_weight * kl_divergence
                    total_influence += weighted_influence
                else:
                    print(
                        f"[WARNING] Attention weight index {idx} out of bounds for {len(attention_weights)} weights"
                    )

            except Exception as e:
                print(f"[WARNING] Influence computation failed for neighbor {j}: {e}")
                continue

        return total_influence

    def select_action(
        self,
        my_features: torch.Tensor,
        neighbor_positions: torch.Tensor,
        item_positions: torch.Tensor,
        agent_idx: int,
    ):
        """Select action using item-aware attention policy."""
        with torch.no_grad():
            logits, value, attention_weights = self.agents[agent_idx](
                my_features, neighbor_positions, item_positions
            )
            probs = F.softmax(logits, dim=-1)

            dist = Categorical(probs)
            action = dist.sample()

            return (
                action.item(),
                dist.log_prob(action).item(),
                value.item(),
                attention_weights,
            )

    def train_episode(self):
        """Modified training with item-aware observations and robust error handling"""
        raw_obs = self.env.reset()
        agent_positions = []
        structured_states = []
        item_positions_all = []

        # Initial processing
        for agent_idx in range(self.num_agents):
            agent_obs = (
                raw_obs[f"player_{agent_idx}"]
                if isinstance(raw_obs, dict)
                else raw_obs[agent_idx]
            )

            # Handle position extraction differently for array vs dict
            if isinstance(agent_obs, dict):
                pos = agent_obs.get("WORLD.AVATAR.POSITION", np.zeros(2))
            else:
                # Assuming position is first 2 elements if array
                pos = agent_obs[:2] if len(agent_obs) >= 2 else np.zeros(2)

            agent_positions.append(pos)

            # Extract item positions for this agent
            item_positions = extract_items_positions(agent_obs)
            item_positions_all.append(item_positions)

            features = extract_structured_features(
                agent_obs, agent_idx, agent_positions
            )
            structured_tensor = torch.FloatTensor(features).to(self.device)
            structured_states.append(structured_tensor)

        episode_reward = 0
        episode_length = 0
        total_attention_entropy = 0
        done = False

        while not done and episode_length < 1000:
            # Get actions for all agents
            actions = []
            log_probs = []
            values = []
            all_attention_weights = []
            neighbor_positions_list = []

            for agent_idx in range(self.num_agents):
                # Get neighbor positions relative to this agent
                my_pos = agent_positions[agent_idx]
                neighbor_positions = []
                neighbor_indices = []

                for j in range(self.num_agents):
                    if j != agent_idx:
                        rel_pos = np.array(agent_positions[j]) - np.array(my_pos)
                        neighbor_positions.append(rel_pos)
                        neighbor_indices.append(j)

                # Convert to tensors with proper shape handling
                if len(neighbor_positions) > 0:
                    neighbor_positions_tensor = torch.FloatTensor(
                        neighbor_positions
                    ).to(self.device)
                else:
                    neighbor_positions_tensor = torch.zeros((0, 2), device=self.device)

                # Handle item positions
                if len(item_positions_all[agent_idx]) > 0:
                    item_positions_tensor = torch.FloatTensor(
                        item_positions_all[agent_idx]
                    ).to(self.device)
                else:
                    item_positions_tensor = torch.zeros((0, 2), device=self.device)

                neighbor_positions_list.append(neighbor_positions_tensor)

                action, log_prob, value, attention_weights = self.select_action(
                    structured_states[agent_idx],
                    neighbor_positions_tensor,
                    item_positions_tensor,
                    agent_idx,
                )

                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)
                all_attention_weights.append((neighbor_indices, attention_weights))

                # Track attention entropy (handle empty attention weights)
                if attention_weights is not None and attention_weights.numel() > 0:
                    # Add small epsilon to prevent log(0)
                    attention_weights_safe = attention_weights + 1e-8
                    attention_entropy = (
                        -(attention_weights_safe * attention_weights_safe.log())
                        .sum()
                        .item()
                    )
                    total_attention_entropy += attention_entropy

            # Environment step
            step_return = self.env.step(actions)
            if len(step_return) == 4:
                next_raw_obs, rewards, done, info = step_return
            else:
                next_raw_obs, rewards, done, truncated, info = step_return
                done = done or truncated

            # Compute influence rewards with error handling
            influence_rewards = []
            for i in range(self.num_agents):
                neighbor_indices, attention_weights = all_attention_weights[i]
                try:
                    influence = self.compute_attention_influence_reward(
                        i,
                        structured_states,
                        neighbor_positions_list,
                        [
                            torch.FloatTensor(item_positions_all[j]).to(self.device)
                            for j in range(self.num_agents)
                        ],
                        neighbor_indices,
                        attention_weights,
                    )
                    influence_rewards.append(influence)
                except Exception as e:
                    print(f"[WARNING] Influence computation failed for agent {i}: {e}")
                    influence_rewards.append(0.0)

            # Combine rewards
            current_influence_weight = self.get_current_influence_weight()
            modified_rewards = [
                rewards[i] + current_influence_weight * influence_rewards[i]
                for i in range(self.num_agents)
            ]

            # Store experience
            self.memory["states"].extend(structured_states)
            self.memory["neighbor_positions"].extend(neighbor_positions_list)
            self.memory["item_positions"].extend(
                [
                    torch.FloatTensor(item_positions_all[i]).to(self.device)
                    for i in range(self.num_agents)
                ]
            )
            self.memory["actions"].extend(actions)
            self.memory["logprobs"].extend(log_probs)
            self.memory["rewards"].extend(modified_rewards)
            self.memory["values"].extend(values)
            self.memory["is_terminals"].extend([done] * self.num_agents)
            self.memory["attention_weights"].extend(
                [all_attention_weights[i][1] for i in range(self.num_agents)]
            )

            # Update for next step
            agent_positions = []
            structured_states = []
            item_positions_all = []

            for agent_idx in range(self.num_agents):
                agent_obs = (
                    next_raw_obs[f"player_{agent_idx}"]
                    if isinstance(next_raw_obs, dict)
                    else next_raw_obs[agent_idx]
                )

                if isinstance(agent_obs, dict):
                    pos = agent_obs.get("WORLD.AVATAR.POSITION", np.zeros(2))
                else:
                    pos = agent_obs[:2] if len(agent_obs) >= 2 else np.zeros(2)

                agent_positions.append(pos)
                item_positions_all.append(extract_items_positions(agent_obs))

                features = extract_structured_features(
                    agent_obs, agent_idx, agent_positions
                )
                structured_tensor = torch.FloatTensor(features).to(self.device)
                structured_states.append(structured_tensor)

            episode_reward += info.get("collective_reward", sum(rewards))
            episode_length += 1

        # Train PPO after episode
        if len(self.memory["states"]) > 0:
            self._update_ppo()

        # Update metrics
        avg_attention_entropy = (
            total_attention_entropy / (episode_length * self.num_agents)
            if episode_length > 0
            else 0
        )
        self.metrics["collective_rewards"].append(episode_reward)
        self.metrics["episode_lengths"].append(episode_length)
        self.metrics["attention_entropy"].append(avg_attention_entropy)
        self.step_count += episode_length
        self.episode_count += 1

        # Log to wandb
        if self.use_wandb:
            wandb.log(
                {
                    "episode_reward": episode_reward,
                    "episode_length": episode_length,
                    "influence_weight": current_influence_weight,
                    "avg_attention_entropy": avg_attention_entropy,
                    "avg_influence_reward": (
                        np.mean(influence_rewards) if influence_rewards else 0
                    ),
                    "episode": self.episode_count,
                    "total_steps": self.step_count,
                }
            )

        return episode_reward

    def _update_ppo(self):
        """Update PPO networks using collected experiences with item-aware attention."""
        # Convert lists to tensors
        states = torch.stack(self.memory["states"]).detach()
        neighbor_positions = self.memory["neighbor_positions"]
        item_positions = self.memory["item_positions"]
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
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + 1e-8
        )

        # Calculate advantages
        advantages = discounted_rewards - values.detach()

        # PPO update for each agent
        total_loss = 0
        for epoch in range(self.k_epochs):
            for agent_idx in range(self.num_agents):
                agent_indices = list(range(agent_idx, len(states), self.num_agents))
                if not agent_indices:
                    continue

                agent_states = states[agent_indices]
                agent_neighbor_positions = [
                    neighbor_positions[i] for i in agent_indices
                ]
                agent_item_positions = [item_positions[i] for i in agent_indices]
                agent_actions = actions[agent_indices]
                agent_old_logprobs = old_logprobs[agent_indices]
                agent_advantages = advantages[agent_indices]
                agent_discounted_rewards = discounted_rewards[agent_indices]

                all_logits = []
                all_values = []
                all_entropies = []

                for i, (state, neighbor_pos, item_pos) in enumerate(
                    zip(agent_states, agent_neighbor_positions, agent_item_positions)
                ):
                    logits, value, _ = self.agents[agent_idx](
                        state, neighbor_pos, item_pos
                    )
                    all_logits.append(logits)
                    all_values.append(value)

                    probs = F.softmax(logits, dim=-1)
                    entropy = -(probs * probs.log()).sum()
                    all_entropies.append(entropy)

                all_logits = torch.stack(all_logits)
                all_values = torch.stack(all_values).squeeze()
                all_entropies = torch.stack(all_entropies)

                dist = Categorical(F.softmax(all_logits, dim=-1))
                new_logprobs = dist.log_prob(agent_actions)

                ratios = torch.exp(new_logprobs - agent_old_logprobs.detach())
                surr1 = ratios * agent_advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * agent_advantages
                )

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(all_values, agent_discounted_rewards)
                entropy_loss = -all_entropies.mean()

                loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
                total_loss += loss.item()

                self.optimizers[agent_idx].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agents[agent_idx].parameters(), 0.5)
                self.optimizers[agent_idx].step()

        self.metrics["losses"].append(total_loss / (self.k_epochs * self.num_agents))
        self.reset_memory()

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


if __name__ == "__main__":
    wandb.init(project="attention-influence-ppo-item-aware")

    num_agents = 5
    feature_dim = 50
    action_dim = 9

    trainer = AttentionInfluencePPO(
        env=None,  # Replace with your environment
        num_agents=num_agents,
        feature_dim=feature_dim,
        action_dim=action_dim,
        influence_weight=0.1,
        use_wandb=True,
    )

    for episode in range(1000):
        reward = trainer.train_episode()

        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {reward}")
            trainer.save_checkpoint(f"checkpoint_item_aware_episode_{episode}.pt")

            if len(trainer.metrics["attention_entropy"]) > 0:
                avg_entropy = np.mean(list(trainer.metrics["attention_entropy"])[-100:])
                wandb.log({"avg_attention_entropy_100ep": avg_entropy})
