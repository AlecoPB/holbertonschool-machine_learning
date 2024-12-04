#!/usr/bin/env python3
"""
This is some documentation
"""
import requests
from datetime import datetime


def fetch_first_launch():
    """
    Display the first launch
    """
    # Fetch all launches
    response = requests.get("https://api.spacexdata.com/v4/launches")

    launches = response.json()

    # Sort launches by date
    launches.sort(key=lambda launch: launch.get("date_unix", float("inf")))

    # Take only the oldest one (the first)
    first_launch = launches[0]

    # Extract launch details
    name = first_launch['name']
    date_local = first_launch['date_local']
    rocket_id = first_launch['rocket']
    launchpad_id = first_launch['launchpad']

    # Fetch rocket and launchpad details
    response = requests.get(f"https://api.spacexdata.com/v4/rockets/{rocket_id}")
    rocket_name = response.json().get("name")

    response = requests.get(f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}")
    launchpad_details = response.json()

    # Format and print the result
    launchpad_name = launchpad_details.get("name")
    launchpad_locality = launchpad_details.get("locality")
    print(f"{name} ({date_local}) {rocket_name} - {launchpad_name} ({launchpad_locality})")


def fetch_rocket_name(rocket_id):
    """
    Fetch the name of the rocket
    """
    if not rocket_id:
        return "Unknown"
    response = requests.get(f"https://api.spacexdata.com/v4/rockets/{rocket_id}")
    rocket_data = response.json()
    return rocket_data.get("name")

def fetch_launchpad_details(launchpad_id):
    """
    Fetch the launchpad details (duh)
    """
    response = requests.get(f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}")
    response.raise_for_status()
    return response.json()

if __name__ == "__main__":
    fetch_first_launch()
