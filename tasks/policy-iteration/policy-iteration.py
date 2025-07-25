import gymnasium as gym
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gamma", default=0.95, type=float, help="Discount factor.")
parser.add_argument("--steps", default=150, type=int, help="Number of value iteration steps to perform.")
parser.add_argument("--max_value_iteration_steps", default=150, type=int, help="Number of value iteration steps to perform.")
parser.add_argument("--render_episodes", default=5, type=int)
parser.add_argument("--evaluate_episodes", default=200, type=int)
parser.add_argument("--max_episode_steps", default=100, type=int)
parser.add_argument("--is_slippery", default=False, type=bool)
parser.add_argument("--map", default="4x4", type=str)

def render_agent(env, policy, args):
    for episode in range(0, args.render_episodes):

        state, info = env.reset()

        for step in range(0, args.max_episode_steps):
            action = policy[state]

            new_state, reward, terminated, truncated, info = env.step(action)

            if (terminated or truncated):
                break

            state = new_state

def evaluate_agent(env, policy, args):
    rewards = []
    for episode in range(0, args.evaluate_episodes):

        state, info = env.reset()

        episode_reward = 0

        # compute received reward for the whole episode
        for step in range(0, args.max_episode_steps):
            action = policy[state]

            new_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if (terminated or truncated):
                break

            state = new_state
        
        rewards.append(episode_reward)
    
    return np.mean(rewards), np.std(rewards)


def get_value_update(env, previous_values, state, policy, gamma):
    """
    V_i[state] = sum(p(new_state | state, policy(state)) * (reward + gamma * V_(i-1)[policy(state)])))
    """
    possible_transitions = env.unwrapped.P[state][policy[state]]

    return get_action_value(possible_transitions, previous_values, gamma)

def get_action_value(possible_transitions, values, gamma):
    return np.sum([
        probability * (reward + gamma * values[new_state])
        for probability, new_state, reward, done in possible_transitions
    ])

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    env = gym.make("FrozenLake-v1", map_name=args.map, is_slippery=args.is_slippery)
    render_env = gym.make("FrozenLake-v1", map_name=args.map, is_slippery=args.is_slippery, render_mode="human")

    policy = np.zeros((env.observation_space.n), dtype=np.int32)

    for step in range(0, args.steps):
        # policy evaluation - value iteration for policy
        values = np.zeros((env.observation_space.n))
        previous_values = np.zeros((env.observation_space.n))

        for vstep in range(0, args.max_value_iteration_steps):
            for state in range(0, env.observation_space.n):
                values[state] = get_value_update(env, previous_values, state, policy, args.gamma)

            if np.allclose(values, previous_values):
                print(f"Value Interation converged at step {vstep}")
                break

            previous_values = values.copy()

        old_policy = policy.copy()
        # policy improvement
        for state in range(0, env.observation_space.n):
            # policy[state] = argmax_a(E_new_state(reward + gamma * value(new_state) | state, a))
            
            # compute value to taking all actions in given state
            action_values = [
                get_action_value(possible_transitions, values, args.gamma) 
                for possible_transitions in env.unwrapped.P[state].values()
            ]
            policy[state] = np.argmax(action_values)
        
        if np.allclose(policy, old_policy):
            print(f"Policy Iteration converged at step {step}")
            break

    mean, std = evaluate_agent(env, policy, args)
    print(f"mean reward {mean} +- {std}")
    
    render_agent(render_env, policy, args)    
