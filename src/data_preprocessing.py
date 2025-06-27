# Data Preprocessing Script
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    df = df.dropna()
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df)
    return df
