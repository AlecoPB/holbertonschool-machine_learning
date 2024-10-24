#!/usr/bin/env python3
import numpy as np


def play(env, Q, max_steps=100):
    """
    Plays and renders a map
    """
    state, _ = env.reset()
    total_reward = 0
    rendered_outputs = []

    for _ in range(max_steps):

        # Render and append
        rendered_output = env.render()
        rendered_outputs.append(rendered_output)

        # Decide and take action
        action = np.argmax(Q[state])  # Exploit
        next_state, reward, done, info, _ = env.step(action)

        # We assign reward each time, since it can only be 1
        total_reward = reward

        state = next_state

        if done:
            break

    # We render one last time
    rendered_output = env.render()
    rendered_outputs.append(rendered_output)

    return total_reward, rendered_outputs
