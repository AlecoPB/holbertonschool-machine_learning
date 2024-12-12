#!/usr/bin/env python3
"""
This is some documentation
"""


def schools_by_topic(mongo_collection, topic):
    """
    return the list of schools having a topic
    """
    return list(mongo_collection.find({'topics': topic}))
