import gymnasium as gym
import random
import numpy as np
import tqdm

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
render_env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")

# Q learning
LEARNING_RATE = 0.7
GAMMA = 0.95
N_TRAIN_EPISODES = 50000
N_EVALUATE_EPISODES = 500
N_RENDER_EPISODES = 5

MAX_STEPS = 100
MIN_EPSILON = 0.01
MAX_EPSILON = 1
DECAY_RATE = 0.0005

def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
  """
  Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
  :param env: The evaluation environment
  :param max_steps: Maximum number of steps per episode
  :param n_eval_episodes: Number of episode to evaluate the agent
  :param Q: The Q-table
  :param seed: The evaluation seed array (for taxi-v3)
  """
  episode_rewards = []
  for episode in tqdm.tqdm(range(n_eval_episodes)):
    if seed:
      state, info = env.reset(seed=seed[episode])
    else:
      state, info = env.reset()
    total_rewards_ep = 0

    for _ in range(max_steps):
      # Take the action (index) that have the maximum expected future reward given that state
      action = greedy_policy(Q, state)
      new_state, reward, terminated, truncated, info = env.step(action)
      total_rewards_ep += reward

      if terminated or truncated:
        break
      
      state = new_state

    episode_rewards.append(total_rewards_ep)
    
  mean_reward = np.mean(episode_rewards)
  std_reward = np.std(episode_rewards)

  return mean_reward, std_reward

def render_agent(env):
    for episode in tqdm.tqdm(range(N_RENDER_EPISODES)):
      state, info = env.reset()

      for _ in range(MAX_STEPS):
        # Take the action (index) that have the maximum expected future reward given that state
        action = greedy_policy(Q, state)
        new_state, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
          break
        
        state = new_state

def epsilon_greedy_policy(Q, env, state, epsilon):
    """
    Chooses action according to epsilon greedy policy. 
     
    Either selects the best action with `1 - epsilon` probability or random action with `epsilon` probability
    """
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return greedy_policy(Q, state)

def greedy_policy(Q, state):
   return np.argmax(Q[state])

# init q table with zeros
Q = np.zeros((env.observation_space.n, env.action_space.n))
    
# iterate for given number of episodes
for episode in tqdm.tqdm(range(N_TRAIN_EPISODES)):

    # reset env
    state, info = env.reset()

    # compute exponentially decayed epsilon
    epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY_RATE * episode)

    # loop until episode ends
    for step in range(MAX_STEPS):
        # select action in env according to e-greedy policy
        action = epsilon_greedy_policy(Q, env, state, epsilon)

        # take action in env
        new_state, reward, terminated, truncated, info = env.step(action)

        # perform q learning temporal difference update 
        # Q[state][action] = Q[state][action] + learning_rate * (reward + gamma * max_a(Q[newstate][a]) - Q[state][action])
        Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[new_state]) - Q[state, action])
        if (terminated or truncated):
            break

        state = new_state

print("Q:", Q)

mean_reward, std_reward = evaluate_agent(env, MAX_STEPS, N_EVALUATE_EPISODES, Q, False)
print(f"Mean reward: {mean_reward:.2f} +- {std_reward:.2f}")

render_agent(render_env)
