#!/usr/bin/env python3
"""
This is some documentation
"""


def list_all(mongo_collection):
    """
    List all documents in a collection
    """
    return list(mongo_collection.find())
