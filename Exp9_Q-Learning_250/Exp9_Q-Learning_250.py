import numpy as np
import gymnasium as gym
game = gym.make("FrozenLake-v1", is_slippery=False)
Q = np.zeros((16, 4))
for round in range(2000):
    state, _ = game.reset()
    done = False
    while not done:
        if np.random.rand() < 0.2:
            move = game.action_space.sample()
        else:
            move = np.random.choice(np.flatnonzero(Q[state] == Q[state].max()))
        new_state, reward, done, _, _ = game.step(move)
        Q[state, move] += 0.8 * (reward + 0.95 * np.max(Q[new_state]) - Q[state, move])
        state = new_state
print("Training done. The agent has learned the path.")
arrows = ["left", "down", "right", "up"]
for square in [0, 6, 10, 14]:
    best = np.argmax(Q[square])
    print("From square", square, "-> best move:", arrows[best])
    goal = 15
    wins = 0
    for game_no in range(100):
        state, _ = game.reset()
        for step in range(100):
            state, reward, end, cut, _ = game.step(np.argmax(Q[state]))
            if end or cut:
                break
        if state == goal:
            wins += 1
    print("Success rate:", wins, "out of 100 games won")
