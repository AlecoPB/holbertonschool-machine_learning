#!/usr/bin/env python3
"""
This is some documentation
"""
import requests


def sentientPlanets():
    """Return a list of planets that hold life

    Returns:
        present: list of planets
    """
    # Initialize list to be returned
    present = []

    for i in range(70):
        # Get current planet
        current_planet = requests.get('https://swapi-api.hbtn.io/api/planets/'
                                    + str(i))

        # Check if residents list is empty
        hosts_life = current_planet.json().get('residents').is_empty()

        # Add to list if life is present
        present.append(current_planet.json().get('name') if hosts_life else None)

    return present
