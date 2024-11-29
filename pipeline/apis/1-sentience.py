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
        # Get current species
        current_species = requests.get('https://swapi-api.hbtn.io/api/species/'
                                    + str(i))

        # Check if residents list is empty
        designation = current_species.json().get('designation')
        sentient = False
        if designation is not None and designation == 'sentient': 
            sentient = True
        # Add to list if life is present
        home_planet = requests.get(current_species.json().get('homeworld'))
        if sentient and home_planet is not None:
            home_planet = home_planet.json().get('homeworld')
            present.append(home_planet.json().get('name'))

    return present
