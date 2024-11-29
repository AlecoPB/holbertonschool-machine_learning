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
        if sentient:
            present.append(current_species.json().get('homeworld'))

    return present
