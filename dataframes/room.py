import pandas as pd

class Room:
    def __init__(self, path):
        self.df = pd.read_csv(path)