#!/usr/bin/env python3
import requests
from collections import Counter

def get_launches_per_rocket():
    launches_url = "https://api.spacexdata.com/v4/launches"
    rockets_url = "https://api.spacexdata.com/v4/rockets"

    # Fetch launches and rockets
    launches = requests.get(launches_url).json()
    rockets = requests.get(rockets_url).json()

    # Map rocket IDs to their names
    rocket_names = {rocket["id"]: rocket["name"] for rocket in rockets}

    # Count launches per rocket ID
    rocket_launch_counts = Counter(launch["rocket"] for launch in launches)

    # Prepare results as (rocket_name, count)
    results = [
        (rocket_names.get(rocket_id, "Unknown"), count)
        for rocket_id, count in rocket_launch_counts.items()
    ]

    # Sort by number of launches (descending) and name (ascending)
    results.sort(key=lambda x: (-x[1], x[0]))

    # Print results
    for rocket_name, count in results:
        print(f"{rocket_name}: {count}")

if __name__ == "__main__":
    get_launches_per_rocket()
