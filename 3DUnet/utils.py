import os


def create_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory
