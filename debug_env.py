#!/usr/bin/env python3
"""Debug script to test MeltingPot environment directly."""

import traceback
import sys
import os

# Add the path to your meltingpot modules
sys.path.append('/home/kaou-internship/meltingpot')

def test_basic_import():
    """Test if basic imports work."""
    print("=== Testing Basic Imports ===")
    try:
        from meltingpot import substrate
        print("✓ MeltingPot substrate import successful")

        from meltingpot.examples.rllib import utils
        print("✓ Utils import successful")

        import ray
        print("✓ Ray import successful")

        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_substrate_creation():
    """Test creating the substrate directly."""
    print("\n=== Testing Substrate Creation ===")
    try:
        from meltingpot import substrate

        # Get the config
        config = substrate.get_config("clean_up")
        print(f"✓ Got substrate config: {config.default_player_roles}")

        # Build the environment
        env = substrate.build("clean_up", roles=config.default_player_roles)
        print(f"✓ Built substrate environment")

        # Test basic properties
        obs_spec = env.observation_spec()
        action_spec = env.action_spec()
        print(f"✓ Environment specs - Obs: {len(obs_spec)}, Action: {len(action_spec)}")

        # Test reset
        timestep = env.reset()
        print(f"✓ Environment reset successful")
        print(f"   Observation keys for player 0: {list(timestep.observation[0].keys())}")

        # Test step
        actions = [0] * len(action_spec)  # All agents do nothing
        timestep = env.step(actions)
        print(f"✓ Environment step successful")
        print(f"   Rewards: {timestep.reward}")
        print(f"   Episode done: {timestep.last()}")

        env.close()
        print("✓ Environment closed successfully")
        return True, len(obs_spec)

    except Exception as e:
        print(f"✗ Substrate creation failed: {e}")
        traceback.print_exc()
        return False, 0

def test_wrapper():
    """Test the MeltingPot wrapper."""
    print("\n=== Testing MeltingPot Wrapper ===")
    try:
        from meltingpot.examples.rllib import utils

        env_config = {
            "substrate": "clean_up",
            "roles": ("default", "default", "default", "default", "default", "default", "default")
        }

        # Create wrapped environment
        env = utils.env_creator(env_config)
        print(f"✓ Wrapper environment created")

        # Check spaces
        print(f"✓ Observation space: {len(env.observation_space.spaces)} agents")
        print(f"✓ Action space: {len(env.action_space.spaces)} agents")

        # Test reset
        obs, info = env.reset()
        print(f"✓ Reset successful - got {len(obs)} observations")
        print(f"   Agent IDs: {list(obs.keys())}")

        # Create action dict
        action_dict = {}
        for agent_id in obs.keys():
            action_dict[agent_id] = 0  # All agents do nothing

        # Test step
        obs, rewards, terminated, truncated, info = env.step(action_dict)
        print(f"✓ Step successful")
        print(f"   Rewards: {rewards}")
        print(f"   Terminated: {terminated}")
        print(f"   Truncated: {truncated}")

        env.close()
        print("✓ Wrapper environment closed successfully")
        return True

    except Exception as e:
        print(f"✗ Wrapper test failed: {e}")
        traceback.print_exc()
        return False

def test_multiple_steps():
    """Test running multiple steps."""
    print("\n=== Testing Multiple Steps ===")
    try:
        from meltingpot.examples.rllib import utils

        env_config = {
            "substrate": "clean_up",
            "roles": ("default", "default", "default", "default", "default", "default", "default")
        }

        env = utils.env_creator(env_config)
        obs, info = env.reset()

        print(f"Running 10 steps...")
        for step in range(10):
            # Random actions
            action_dict = {}
            for agent_id in obs.keys():
                action_dict[agent_id] = step % 9  # Cycle through actions 0-8

            obs, rewards, terminated, truncated, info = env.step(action_dict)

            if terminated.get('__all__', False):
                print(f"Episode ended at step {step}")
                break

            if step % 5 == 0:
                print(f"Step {step}: Agent rewards = {list(rewards.values())}")

        env.close()
        print("✓ Multiple steps test successful")
        return True

    except Exception as e:
        print(f"✗ Multiple steps test failed: {e}")
        traceback.print_exc()
        return False

def test_ray_registration():
    """Test Ray environment registration."""
    print("\n=== Testing Ray Environment Registration ===")
    try:
        import ray
        from ray import tune
        from meltingpot.examples.rllib import utils

        # Initialize Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Register environment
        tune.register_env("meltingpot", utils.env_creator)
        print("✓ Environment registered with Ray")

        # Test creating through Ray
        env_config = {
            "substrate": "clean_up",
            "roles": ("default", "default", "default", "default", "default", "default", "default")
        }

        env = tune.registry.env_registry.get("meltingpot")(env_config)
        print("✓ Environment created through Ray registry")

        # Quick test
        obs, info = env.reset()
        print(f"✓ Reset through Ray successful - {len(obs)} agents")

        env.close()
        ray.shutdown()
        print("✓ Ray registration test successful")
        return True

    except Exception as e:
        print(f"✗ Ray registration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 MeltingPot Environment Debugging Script")
    print("=" * 50)

    tests = [
        ("Basic Imports", test_basic_import),
        ("Substrate Creation", test_substrate_creation),
        ("Wrapper Test", test_wrapper),
        ("Multiple Steps", test_multiple_steps),
        ("Ray Registration", test_ray_registration),
    ]

    results = []
    for name, test_func in tests:
        try:
            if name == "Substrate Creation":
                success, num_agents = test_func()
                results.append((name, success))
                if success:
                    print(f"   Number of agents: {num_agents}")
            else:
                success = test_func()
                results.append((name, success))
        except Exception as e:
            print(f"✗ {name} crashed: {e}")
            results.append((name, False))

    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"   {name}: {status}")

    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 All tests passed! Environment should work with RLlib.")
    else:
        print("\n❌ Some tests failed. Fix these issues before using with RLlib.")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
