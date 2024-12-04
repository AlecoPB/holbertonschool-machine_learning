#!/usr/bin/env python3
"""
This is some documentation
"""
import sys
import requests
from datetime import datetime


def get_user_location(api_url):
    """Shows location of a user

    Args:
        api_url (string): url of user
    """
    try:
        # Make a GET request to the GitHub API
        response = requests.get(api_url)

        # If user isn't found
        if response.status_code == 404:
            print("Not found")

        elif response.status_code == 403:
            # Handle rate limit exceeded
            reset_time = response.headers.get("X-RateLimit-Reset")
            if reset_time:
                reset_time = int(reset_time)
                reset_in_minutes = (datetime.fromtimestamp(reset_time)
                                    - datetime.now()).total_seconds() / 60
                print(f"Reset in {int(reset_in_minutes)} min")
            else:
                print("Reset time unavailable")

        elif response.status_code == 200:
            user_data = response.json()
            # Check if the location field exists
            location = user_data.get("location")

            if location:
                print(location)
            else:
                print("Location not found")


        else:
            print(f"Unexpected status code: {response.status_code}")
    except requests.RequestException as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    # Ensure the script receives the correct number of arguments
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <GitHub API URL>")
    else:
        api_url = sys.argv[1]
        get_user_location(api_url)
