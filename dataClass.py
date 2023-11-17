import pandas as pd

# should rename this file to match the class
class DataSet:
    def __init__(self):
        self.df = pd.DataFrame()
        self.model = ""
        self.matrix = ""
        self.dict_index = {}
        self.dict_values = {}
