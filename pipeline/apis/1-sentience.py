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
        s_class = current_species.json().get('class') 
        sentient = False
        if (designation is not None and designation == 'sentient')\
            or (s_class is not None and s_class == 'sentient'): 
            sentient = True
        # Add to list if life is present
        home_planet = current_species.json().get('homeworld')
        if sentient and home_planet is not None:
            home_planet = requests.get(home_planet).json().get('name')
            present.append(home_planet)

    return present
