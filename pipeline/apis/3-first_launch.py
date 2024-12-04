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
    api_url = "https://api.spacexdata.com/v4/launches"

    # Fetch all launches
    response = requests.get(api_url)
    response.raise_for_status()  # Raise an exception for HTTP errors

    launches = response.json()
    if not launches:
        print("No launches found")
        return

    # Sort launches by date_unix (ascending)
    launches.sort(key=lambda launch: launch.get("date_unix", float("inf")))

    # Select the first launch
    first_launch = launches[0]

    # Extract launch details
    name = first_launch.get("name", "Unknown")
    date_unix = first_launch.get("date_unix")
    date_local = datetime.fromtimestamp(date_unix).strftime("%Y-%m-%d %H:%M:%S") if date_unix else "Unknown"
    rocket_id = first_launch.get("rocket")
    launchpad_id = first_launch.get("launchpad")

    # Fetch rocket and launchpad details
    rocket_name = fetch_rocket_name(rocket_id)
    launchpad_details = fetch_launchpad_details(launchpad_id)

    # Format and print the result
    launchpad_name = launchpad_details.get("name", "Unknown")
    launchpad_locality = launchpad_details.get("locality", "Unknown")
    print(f"{name} ({date_local}) {rocket_name} - {launchpad_name} ({launchpad_locality})")


def fetch_rocket_name(rocket_id):
    if not rocket_id:
        return "Unknown"
    try:
        response = requests.get(f"https://api.spacexdata.com/v4/rockets/{rocket_id}")
        response.raise_for_status()
        rocket_data = response.json()
        return rocket_data.get("name", "Unknown")
    except requests.RequestException:
        return "Unknown"

def fetch_launchpad_details(launchpad_id):
    if not launchpad_id:
        return {"name": "Unknown", "locality": "Unknown"}
    try:
        response = requests.get(f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}")
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return {"name": "Unknown", "locality": "Unknown"}

if __name__ == "__main__":
    fetch_first_launch()
