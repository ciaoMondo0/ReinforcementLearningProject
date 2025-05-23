import gymnasium as gym
import numpy as np
import pickle

def run(episodes, is_training=True, render=False):
    env = gym.make(
        'FrozenLake-v1',
        map_name="8x8",
        is_slippery=True,
        render_mode='human' if render else None
    )

    if is_training:
        q_table = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        with open('frozen_lake8x8.pkl', 'rb') as f:
            q_table = pickle.load(f)

    learning_rate = 0.9
    discount_factor = 0.9
    epsilon = 1.0
    epsilon_decay_rate = 0.0001
   np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for episode in range(episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False

        while True:
            if is_training and np.random.default_rng() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])

            next_state, reward, terminated, truncated, _ = env.step(action)

            if is_training:
                q_table[state, action] += learning_rate * (
                    reward + discount_factor * np.max(q_table[next_state, :]) - q_table[state, action]
                )

            state = next_state

            if terminated or truncated:
                break

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if epsilon == 0:
            learning_rate = 0.0001

        if reward == 1:
            rewards_per_episode[episode] = 1

    env.close()

    if is_training:
        with open("frozen_lake8x8.pkl", "wb") as f:
            pickle.dump(q_table, f)


if __name__ == '__main__':
    run(episodes=12000, is_training=True, render=False)

    run(episodes=100, is_training=False, render=True)

 
