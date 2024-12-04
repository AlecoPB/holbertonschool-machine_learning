
#!/usr/bin/env python3
"""
APIs project
By Ced
"""
import sys
import time
import requests


def user_location(argument):
    """
    This is a function that takes in a string argument
    and returns the location of the user
    no main.py is this exercice
    """
    # print(f"Voici l'argument passé : {argument}")
    r = requests.get(argument)
    # print(f"Voici le code de retour : {r.status_code}")

    # if error 403
    if r.status_code == 403:
        X = r.headers.get("X-RateLimit-Reset")
        wait = int(X) - int(time.time())
        wait_minute = wait // 60
        print(f"Reset in {wait_minute} min")

    # if error 404, wrong url
    elif r.status_code == 404:
        print(f"Not found")

    elif r.status_code == 200:
        loc = r.json().get("location")
        print(loc)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Erreur : Un unique argument est requis.")

    else:
        user_location(sys.argv[1])  # Le premier argument est `sys.argv[1]`
