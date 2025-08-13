#!/usr/bin/env python3
"""
Self-play training script for item-aware attention social influence models.
FINAL VERSION - All issues resolved.
"""

import os
import argparse
import torch
import numpy as np
import wandb
from datetime import datetime
from typing import Dict, Any, List

# Import our custom PPO implementations
from examples.rllib.influence_ppo import SocialInfluencePPO
from examples.rllib.attention_influence_ppo import (
    AttentionInfluencePPO,
    extract_items_positions,
)

# MeltingPot imports
from meltingpot import substrate
import dmlab2d


def extract_structured_features(obs, agent_idx=None, all_agent_positions=None):
    """Final robust feature extractor that handles all cases."""
    BASE_SCALAR_SIZE = 20
    MAX_OTHER_AGENTS = 4
    FEATURE_SIZE = BASE_SCALAR_SIZE + MAX_OTHER_AGENTS * 2
    features = [0.0] * FEATURE_SIZE

    if isinstance(obs, dict):
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
        obs_size = min(len(obs), BASE_SCALAR_SIZE)
        features[:obs_size] = obs[:obs_size].tolist()

    result = np.array(features, dtype=np.float32)

    # Ensure correct size
    if len(result) != FEATURE_SIZE:
        if len(result) > FEATURE_SIZE:
            result = result[:FEATURE_SIZE]
        else:
            padding = np.zeros(FEATURE_SIZE - len(result))
            result = np.concatenate([result, padding])

    return result


def get_substrate_player_requirements(substrate_name: str) -> Dict[str, Any]:
    """Get player count requirements for different substrates."""
    substrate_requirements = {
        "coins": {"min_players": 2, "max_players": 2, "default": 2},
        "clean_up": {"min_players": 1, "max_players": 15, "default": 7},
        "harvest": {"min_players": 1, "max_players": 15, "default": 5},
        "commons_harvest__open": {"min_players": 1, "max_players": 15, "default": 7},
        "collaborative_cooking": {"min_players": 2, "max_players": 8, "default": 4},
        "territory": {"min_players": 2, "max_players": 8, "default": 4},
        "daycare": {"min_players": 2, "max_players": 8, "default": 6},
        "prisoners_dilemma_in_the_matrix__repeated": {
            "min_players": 2,
            "max_players": 2,
            "default": 2,
        },
        "stag_hunt_in_the_matrix__arena": {
            "min_players": 2,
            "max_players": 2,
            "default": 2,
        },
    }

    if substrate_name in substrate_requirements:
        return substrate_requirements[substrate_name]

    for substrate_pattern, requirements in substrate_requirements.items():
        if substrate_pattern in substrate_name or substrate_name in substrate_pattern:
            return requirements

    return {"min_players": 1, "max_players": 15, "default": 7}


def validate_and_fix_player_count(substrate_name: str, requested_players: int) -> int:
    """Validate and fix player count based on substrate requirements."""
    requirements = get_substrate_player_requirements(substrate_name)

    min_players = requirements["min_players"]
    max_players = requirements["max_players"]
    default_players = requirements["default"]

    if requested_players < min_players:
        print(
            f"[WARNING] Substrate '{substrate_name}' requires at least {min_players} players."
        )
        print(
            f"[WARNING] Changing from {requested_players} to {default_players} players."
        )
        return default_players
    elif requested_players > max_players:
        print(
            f"[WARNING] Substrate '{substrate_name}' supports at most {max_players} players."
        )
        print(f"[WARNING] Changing from {requested_players} to {max_players} players.")
        return max_players
    else:
        print(
            f"[INFO] Substrate '{substrate_name}' supports {requested_players} players."
        )
        return requested_players


class ItemAwareStructuredObsWrapper:
    """Final wrapper with all fixes applied."""

    def __init__(self, env: dmlab2d.Environment):
        self.env = env
        self.num_players = len(env.observation_spec())

        # Calculate feature dimension
        dummy_timestep = env.reset()
        dummy_obs = dummy_timestep.observation[0]
        dummy_features = extract_structured_features(dummy_obs)
        self.feature_dim = len(dummy_features)

        print(
            f"[INFO] Wrapper initialized - Feature dim: {self.feature_dim}, Players: {self.num_players}"
        )

    def reset(self):
        """Reset environment and return dictionary observations."""
        timestep = self.env.reset()
        obs_dict = {}
        for i in range(self.num_players):
            player_key = f"player_{i}"
            obs_dict[player_key] = timestep.observation[i]
        return obs_dict

    def step(self, actions: List[int]):
        """Step environment and return dictionary observations."""
        # Ensure correct action count
        if len(actions) != self.num_players:
            if len(actions) < self.num_players:
                actions = actions + [0] * (self.num_players - len(actions))
            else:
                actions = actions[: self.num_players]

        timestep = self.env.step(actions)

        obs_dict = {}
        for i in range(self.num_players):
            player_key = f"player_{i}"
            obs_dict[player_key] = timestep.observation[i]

        rewards = timestep.reward
        done = timestep.last()

        # Ensure correct reward count
        if len(rewards) != self.num_players:
            if len(rewards) < self.num_players:
                rewards = list(rewards) + [0.0] * (self.num_players - len(rewards))
            else:
                rewards = rewards[: self.num_players]

        info = {"collective_reward": sum(rewards), "individual_rewards": rewards}

        return obs_dict, rewards, done, info

    def close(self):
        """Close the environment."""
        self.env.close()


def create_meltingpot_env(substrate_name: str, num_players: int = None):
    """Create environment with automatic player count validation."""
    # Validate and fix player count
    if num_players is None:
        requirements = get_substrate_player_requirements(substrate_name)
        num_players = requirements["default"]
        print(
            f"[INFO] Using default player count for '{substrate_name}': {num_players}"
        )
    else:
        num_players = validate_and_fix_player_count(substrate_name, num_players)

    # Get substrate config
    config = substrate.get_config(substrate_name)
    valid_roles = list(config.valid_roles)

    if not valid_roles:
        raise ValueError(f"No valid roles found for substrate: {substrate_name}")

    if "default" in valid_roles:
        player_roles = ["default"] * num_players
    else:
        player_roles = [valid_roles[i % len(valid_roles)] for i in range(num_players)]

    print(f"[DEBUG] Using roles: {player_roles}")

    try:
        env = substrate.build(substrate_name, roles=player_roles)
        print(
            f"[INFO] Successfully created environment '{substrate_name}' with {num_players} players"
        )
        wrapped_env = ItemAwareStructuredObsWrapper(env)
        return wrapped_env
    except Exception as e:
        print(f"[ERROR] Failed to create environment '{substrate_name}': {e}")
        raise


def determine_action_dim(substrate_name: str) -> int:
    """Determine action dimension based on substrate."""
    action_dims = {
        "coins": 7,
        "clean_up": 9,
        "harvest": 8,
        "commons_harvest__open": 8,
        "collaborative_cooking": 9,
        "territory": 8,
        "prisoners_dilemma_in_the_matrix__repeated": 2,
        "stag_hunt_in_the_matrix__arena": 2,
        "chicken_in_the_matrix__arena": 2,
        "daycare": 9,
        "boat_race": 8,
        "running_with_scissors": 9,
    }

    if substrate_name in action_dims:
        return action_dims[substrate_name]

    for substrate_pattern, action_dim in action_dims.items():
        if substrate_pattern in substrate_name:
            return action_dim

    print(f"[WARNING] Unknown substrate '{substrate_name}', using default action_dim=8")
    return 8


def create_trainer(config: Dict[str, Any], env):
    """Create the appropriate trainer based on model type."""
    config["action_dim"] = determine_action_dim(config["substrate_name"])
    config["feature_dim"] = env.feature_dim

    print(f"[CONFIG] Creating {config['model_type']} trainer...")
    print(
        f"[CONFIG] Feature dim: {config['feature_dim']}, Action dim: {config['action_dim']}"
    )

    if config["model_type"] == "basic_influence":
        trainer = SocialInfluencePPO(
            env=env,
            num_agents=config["num_players"],
            input_dim=config["feature_dim"],
            action_dim=config["action_dim"],
            influence_weight=config["influence_weight"],
            curriculum_steps=config["curriculum_steps"],
            lr=config["lr"],
            gamma=config["gamma"],
            eps_clip=config["eps_clip"],
            k_epochs=config["k_epochs"],
            device=config["device"],
            use_wandb=config["use_wandb"],
        )
    elif config["model_type"] == "attention_influence":
        trainer = AttentionInfluencePPO(
            env=env,
            num_agents=config["num_players"],
            feature_dim=config["feature_dim"],
            action_dim=config["action_dim"],
            influence_weight=config["influence_weight"],
            curriculum_steps=config["curriculum_steps"],
            lr=config["lr"],
            gamma=config["gamma"],
            eps_clip=config["eps_clip"],
            k_epochs=config["k_epochs"],
            device=config["device"],
            use_wandb=config["use_wandb"],
        )
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")

    return trainer


def get_training_config(args):
    """Get training configuration based on arguments."""
    return {
        "substrate_name": args.substrate,
        "num_players": args.num_players,
        "model_type": args.model_type,
        "feature_dim": None,
        "action_dim": None,
        "hidden_dim": args.hidden_dim,
        "num_episodes": args.num_episodes,
        "lr": args.lr,
        "gamma": args.gamma,
        "eps_clip": args.eps_clip,
        "k_epochs": args.k_epochs,
        "influence_weight": args.influence_weight,
        "curriculum_steps": args.curriculum_steps,
        "device": args.device,
        "save_interval": args.save_interval,
        "log_interval": args.log_interval,
        "use_wandb": args.use_wandb,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
    }


def setup_wandb(config: Dict[str, Any], run_name: str = None):
    """Initialize Weights & Biases logging."""
    if not config["use_wandb"]:
        return None

    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{config['model_type']}_{config['substrate_name']}_{timestamp}"

    wandb.init(
        project=config["wandb_project"],
        entity=config["wandb_entity"],
        name=run_name,
        config=config,
    )
    return run_name


def train_model(config: Dict[str, Any], run_name: str = None,  resume_from: str = None):
    """Main training function."""
    print("=" * 80)
    print(f"Starting training: {config['model_type']} on {config['substrate_name']}")
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
    print("=" * 80)

    # Create environment
    env = create_meltingpot_env(
        substrate_name=config["substrate_name"], num_players=config["num_players"]
    )

    config["num_players"] = env.num_players
    print(f"[CONFIG] Updated player count: {config['num_players']}")

    # Create trainer
    trainer = create_trainer(config, env)
    start_episode = 0
    if resume_from and os.path.exists(resume_from):
        print(f"Loading checkpoint from: {resume_from}")
        trainer.load_checkpoint(resume_from)
        start_episode = trainer.episode_count
        print(f"Resumed from episode: {start_episode}")
    elif resume_from:
        print(f"Warning: Checkpoint file not found: {resume_from}")
        print("Starting from scratch...")

    # Training loop
    print(f"Starting training for {config['num_episodes']} episodes...")
    best_reward = float("-inf")
    episode_rewards = []
    consecutive_errors = 0
    max_consecutive_errors = 10

    for episode in range(start_episode, config['num_episodes']):  # Start from saved episode
        try:
            episode_reward = trainer.train_episode()
            episode_rewards.append(episode_reward)
            consecutive_errors = 0

            # Logging
            if episode % config["log_interval"] == 0:
                avg_reward = (
                    np.mean(episode_rewards[-100:])
                    if len(episode_rewards) >= 100
                    else np.mean(episode_rewards)
                )
                print(
                    f"Episode {episode:6d} | Avg Reward: {avg_reward:8.2f} | Current: {episode_reward:8.2f}"
                )

                if config["use_wandb"]:
                    log_dict = {
                        "episode": episode,
                        "episode_reward": episode_reward,
                        "avg_reward_100": avg_reward,
                        "current_influence_weight": trainer.get_current_influence_weight(),
                    }

                    if (
                        hasattr(trainer, "metrics")
                        and "attention_entropy" in trainer.metrics
                    ):
                        if len(trainer.metrics["attention_entropy"]) > 0:
                            log_dict["attention_entropy"] = list(
                                trainer.metrics["attention_entropy"]
                            )[-1]

                    wandb.log(log_dict)

            # Save checkpoints
            if episode % config["save_interval"] == 0 and episode > 0:
                checkpoint_path = f"checkpoints/{run_name}_episode_{episode}.pt"
                os.makedirs("checkpoints", exist_ok=True)
                trainer.save_checkpoint(checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

                if episode_reward > best_reward:
                    best_reward = episode_reward
                    best_path = f"checkpoints/{run_name}_best.pt"
                    trainer.save_checkpoint(best_path)
                    print(f"New best model saved: {best_path}")

        except Exception as e:
            consecutive_errors += 1
            print(f"[ERROR] Episode {episode} failed: {e}")

            if consecutive_errors >= max_consecutive_errors:
                print(
                    f"[FATAL] Too many consecutive errors ({consecutive_errors}), stopping training"
                )
                break
            else:
                print(
                    f"[RECOVERY] Continuing training... ({consecutive_errors}/{max_consecutive_errors} errors)"
                )
                continue

    # Final save
    if episode_rewards:
        final_path = f"checkpoints/{run_name}_final.pt"
        trainer.save_checkpoint(final_path)
        print(f"Final model saved: {final_path}")

    env.close()
    return trainer, episode_rewards


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Train item-aware attention social influence models"
    )
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Environment settings
    parser.add_argument(
        "--substrate", type=str, default="clean_up", help="MeltingPot substrate name"
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=None,
        help="Number of players (None for substrate default)",
    )

    # Model settings
    parser.add_argument(
        "--model-type",
        type=str,
        default="attention_influence",
        choices=["basic_influence", "attention_influence"],
        help="Type of model to train",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=256, help="Hidden dimension for networks"
    )

    # Training settings
    parser.add_argument(
        "--num-episodes", type=int, default=5000, help="Number of training episodes"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--eps-clip", type=float, default=0.2, help="PPO clipping parameter"
    )
    parser.add_argument("--k-epochs", type=int, default=4, help="PPO update epochs")

    # Social influence settings
    parser.add_argument(
        "--influence-weight",
        type=float,
        default=0.1,
        help="Weight for social influence reward",
    )
    parser.add_argument(
        "--curriculum-steps",
        type=int,
        default=1000000,
        help="Steps for influence curriculum (0 to disable)",
    )

    # System settings
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=500,
        help="Episode interval for saving checkpoints",
    )
    parser.add_argument(
        "--log-interval", type=int, default=10, help="Episode interval for logging"
    )

    # Wandb settings
    parser.add_argument(
        "--use-wandb", action="store_true", help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="item-aware-social-influence",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="Wandb entity name"
    )

    args = parser.parse_args()
    config = get_training_config(args)
    run_name = setup_wandb(config)

    print("Training Configuration:")
    print("-" * 40)
    for key, value in config.items():
        print(f"{key:20s}: {value}")
    print("-" * 40)

    try:
        trainer, rewards = train_model(config, run_name, resume_from=args.resume_from)
        if rewards:
            print(
                f"Training completed! Final average reward: {np.mean(rewards[-100:]):.2f}"
            )
        else:
            print("Training completed but no episodes finished successfully.")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    finally:
        if config["use_wandb"]:
            wandb.finish()


if __name__ == "__main__":
    main()
