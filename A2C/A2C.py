from CarlaEnv import CarlaEnv
import gymnasium

from stable_baselines3 import A2C

# env = gymnasium.make('CartPole-v1')
env = CarlaEnv()

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)
env.close()
model.save("Carla")

del model # remove to demonstrate saving and loading

# model = DQN.load("deepq_cartpole")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()