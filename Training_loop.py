import gym
import gym_dinorun

from stable_baselines3 import PPO


def main():
    env = gym.make("DinoRun-v0")

    model = PPO.load('Dino_PPO_670000', env=env)
    print('Agent created!')

    print("Loop started")
    for i in range(680000, 20000000, 100000):
        print(i)
        # Train the agent
        model.learn(total_timesteps=100000)
        print("Finished training!")
        # Save the agent
        print("Saving model")
        model.save(f"Dino_PPO_{i}")
        print('Model saved!')
        env.reset()
        print('Loading new model')
        model = PPO.load(f'Dino_PPO_{i}', env=env)

    env.reset()
    #   for _ in range(1000):  # Number of "st eps": Make steps large for training.
    #        # env.render()
    #        env.step(env.action_space.sample())  # take a random action
    print("Closing Environment")
    env.close()


if __name__ == '__main__':
    main()