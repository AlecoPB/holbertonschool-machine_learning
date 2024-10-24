#!/usr/bin/env python3
import numpy as np


def play(env, Q, max_steps=100):
    state = env.reset()
    total_reward = 0
    rendered_outputs = []

    for _ in range(max_steps):
        rendered_output = env.render
        rendered_outputs.append(rendered_output)

        action = np.argmax(Q[state])  # Exploit
        next_state, reward, done, info = env.step(action)

        total_reward = reward

        state = next_state

    rendered_output = env.render()
    rendered_outputs.append(rendered_output)

    return total_reward, rendered_outputs
