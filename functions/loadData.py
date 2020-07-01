import pandas as pd 

def loadDf(path):
    df = pd.read_csv(path)
    return df