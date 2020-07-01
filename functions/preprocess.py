from sklearn.preprocessing import normalize

def handleMissing(df):
    df.replace('?',-99999, inplace = True)
    return df

def dropCol(df,headers):
    df.drop(headers, 1, inplace = True)
    return df

def normalizeData(data):
    normalized = normalize(data, axis=1)
    return normalized