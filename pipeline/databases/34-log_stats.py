#!/usr/bin/env python3
"""
This is some documentation
"""
from pymongo import MongoClient

def log_stats():
    """
    Connects to the MongoDB `logs` database, retrieves statistics
    from the `nginx` collection, and prints them in the specified format.
    """
    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:27017/')
    db = client.logs
    collection = db.nginx

    # Total number of logs
    total_logs = collection.count_documents({})
    print(f"{total_logs} logs")

    # Methods statistics
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    print("Methods:")
    for method in methods:
        count = collection.count_documents({"method": method})
        print(f"\tmethod {method}: {count}")

    # Logs with method=GET and path=/status
    get_status_logs = collection.count_documents({"method": "GET", "path": "/status"})
    print(f"{get_status_logs} status check logs")

if __name__ == "__main__":
    log_stats()