from torchrl.envs import StepCounter, TransformedEnv
from torchrl.envs import GymEnv
from torchrl.envs import step_mdp

env = GymEnv("Pendulum-v1")

reset = env.reset()
print('reset', reset)

reset_with_action = env.rand_action(reset)
print('reset_with_action', reset_with_action)

stepped_data = env.step(reset_with_action)
print('stepped_data', stepped_data)

data = step_mdp(stepped_data)
print('data', data)

rollout = env.rollout(max_steps=10)
print('rollout', rollout)

transition = rollout[3]
print('transition', transition)

transformed_env = TransformedEnv(env, StepCounter(max_steps=10))
rollout = transformed_env.rollout(max_steps=100)
print('transformed rollout', rollout)
