import gymnasium as gym
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gamma", default=0.95, type=float, help="Discount factor.")
parser.add_argument("--steps", default=150, type=int, help="Number of value iteration steps to perform.")
parser.add_argument("--render_episodes", default=5, type=int, help="Number of value iteration steps to perform.")
parser.add_argument("--evaluate_episodes", default=200, type=int, help="Number of value iteration steps to perform.")
parser.add_argument("--max_episode_steps", default=100, type=int, help="Number of value iteration steps to perform.")
parser.add_argument("--is_slippery", default=False, type=bool, help="Number of value iteration steps to perform.")
parser.add_argument("--map", default="4x4", type=str, help="Number of value iteration steps to perform.")

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

def compute_qvalue(env, previous_qvalues, state, action, gamma):
    """
    Q_i[state, action] = sum_next_state(reward + gamma * max(Q_(i-1)[next_state]))]
    """
    return np.sum([
        probability * (reward + gamma * np.max(previous_qvalues[new_state]))
        for probability, new_state, reward, done in env.unwrapped.P[state][action]
    ])

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    env = gym.make("FrozenLake-v1", map_name=args.map, is_slippery=args.is_slippery)
    render_env = gym.make("FrozenLake-v1", map_name=args.map, is_slippery=args.is_slippery, render_mode="human")
    
    # print(env.unwrapped.P[0])

    # create initial state values
    qvalues = np.zeros((env.observation_space.n, env.action_space.n))
    previous_qvalues = np.zeros((env.observation_space.n, env.action_space.n))

    # loop over max_steps (horizon) until convergence (resp. to max steps)
    for step in range(0, args.steps):

        # update value for all states
        for state in range(0, env.observation_space.n):
            for action in range(0, env.action_space.n):
                qvalues[state, action] = compute_qvalue(env, previous_qvalues, state, action, args.gamma)
        
        # check for convergence
        if np.allclose(qvalues, previous_qvalues):
            print(f"Converged at step {step}")
            break
        
        previous_qvalues = qvalues.copy()

    # find optimal policy based on state values
    # optimal policy for state is computing reward + next_state values for all actions and choosing the best
    policy = [np.argmax(qvalues[state]) for state in range(env.observation_space.n)]

    mean, std = evaluate_agent(env, policy, args)
    print(f"mean reward {mean} +- {std}")
    
    render_agent(render_env, policy, args)