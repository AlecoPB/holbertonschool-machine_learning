#!/usr/bin/env python3
"""
What even is this task
"""


exit = ["exit", "quit", "goodbye", "bye"]
while True:
    Q = input("Q: ")
    if Q.lower() not in exit:
        print("A:")

    else:
        print("A: Goodbye")
        break
