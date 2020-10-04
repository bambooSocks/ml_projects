import pandas as pd
import numpy as np

df = pd.read_csv("heart.csv")
cols = ['age', 'sex', 'chol', 'restecg', 'thalach', 'oldpeak', 'slope', 'thal', 'target']

data = df[cols].to_numpy()


