#!/usr/bin/env python3
"""
This is some documentation
"""
import requests


def availableShips(passengerCount):
    """Return a list of ships that can carry enough passengers

    Args:
        passengerCount (int): minimum number of passengers

    Returns:
        ship_n: list of ships available
    """
    # Initialize list to be returned
    ship_n = []

    for i in range(70):
        # Get current ship
        current_ship = requests.get('https://swapi-api.hbtn.io/api/starships/' + str(i))

        # Set the name and passenger number of the current ship        
        c_passengers = current_ship.json().get('passengers')
        c_name = current_ship.json().get('name')

        # Convert passengers to int and compare to the minimum
        if c_passengers is not None:
            c_passengers = int(c_passengers.replace(",", ""))
        if c_passengers >= passengerCount:
            ship_n.append(c_name)

    return ship_n
