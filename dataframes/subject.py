import pandas as pd

class Subject:
    def __init__(self, path):
        self.df = pd.read_csv(path)