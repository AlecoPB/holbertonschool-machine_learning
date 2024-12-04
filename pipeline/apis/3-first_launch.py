#!/usr/bin/env python3
"""
This is some documentation
"""
import requests


def fetch_first_launch():
    """
    Display the first launch
    """
    # Fetch all launches
    response = requests.get("https://api.spacexdata.com/v4/launches/upcoming")

    launches = response.json()

    first_launch = min(launches, key=lambda x: x['date_unix'])

    # Extract launch details
    name = first_launch['name']
    date_local = first_launch['date_local']
    rocket_id = first_launch['rocket']
    launchpad_id = first_launch['launchpad']

    # Fetch rocket and launchpad details
    rocket_response = requests.get(f"https://api.spacexdata.com/v4/rockets/{rocket_id}")
    rocket_name = rocket_response.json()['name']

    pad_response = requests.get(f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}")
    launchpad_details = pad_response.json()

    # Format and print the result
    launchpad_name = launchpad_details['name']
    launchpad_locality = launchpad_details['locality']

    # Print the formatted information
    print(f"{name} ({date_local}) {rocket_name} - {launchpad_name}"
          f" ({launchpad_locality})")


if __name__ == "__main__":
    fetch_first_launch()
