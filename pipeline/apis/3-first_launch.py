#!/usr/bin/env python3
"""
APIs project
By Ced
"""
import requests


def main():
    """
    main function
    """
    r = requests.get('https://api.spacexdata.com/v5/launches/')
    launches = r.json()
    for i in range(len(launches)):

        if launches[i].get('name') == "Galaxy 33 (15R) & 34 (12R)":
            launch_name = launches[i].get('name')
            date = launches[i].get('date_local')

            rocket_id = launches[i].get('rocket')
            launcher_id = launches[i].get('launchpad')

    r2 = requests.get('https://api.spacexdata.com/v4/rockets/')
    rockets = r2.json()
    for i in range(len(rockets)):
        if rockets[i].get('id') == rocket_id:
            rocket_name = rockets[i].get('name')

    r3 = requests.get('https://api.spacexdata.com/v4/launchpads/')
    launchpads = r3.json()
    for i in range(len(launchpads)):
        if launchpads[i].get('id') == launcher_id:
            launchpad_name = launchpads[i].get('name')
            launchpad_loc = launchpads[i].get('locality')

    print(f"{launch_name} ({date}) {rocket_name} "
          f"- {launchpad_name} ({launchpad_loc})")


if __name__ == "__main__":

    main()