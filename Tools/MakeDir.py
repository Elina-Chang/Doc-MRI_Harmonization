import os


def MakeDir(*pathes):
    for path in pathes:
        if not os.path.exists(path):
            os.makedirs(path)
