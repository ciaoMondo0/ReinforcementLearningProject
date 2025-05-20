import gymnasium as gym
import numpy as np

alpha, gamma = 0.8, 0.95
epsilon, decay, min_eps = 1.0, 0.995, 0.01
episodes, max_steps = 5000, 100

env = gym.make('FrozenLake-v1', is_slippery=False)
nS, nA = env.observation_space.n, env.action_space.n
Q = np.zeros((nS, nA))

for ep in range(episodes):
    state, _ = env.reset()
    for _ in range(max_steps):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        next_state, reward, term, trunc, _ = env.step(action)
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )
        state = next_state
        if term or trunc:
            break
    epsilon = max(epsilon * decay, min_eps)

success = 0
runs = 100
for _ in range(runs):
    state, _ = env.reset()
    for _ in range(max_steps):
        action = np.argmax(Q[state])
        state, reward, term, trunc, _ = env.step(action)
        if term or trunc:
            success += reward
            break

print(f"Success rate: {success/runs:.2f}")