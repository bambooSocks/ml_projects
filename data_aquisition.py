import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("heart.csv")

X_wo = df[:-1].to_numpy()

y_attr = 'target'
cont_attr = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
sel_attr = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope']

X_sel_wo = df[sel_attr].to_numpy()
X_cont_wo = df[cont_attr].to_numpy()

# Removing outliers from selected variables (based on box plots from data_visualization)
for col in sel_attr:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1    # IQR is interquartile range.
    _filter = (df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)
    df = df.loc[_filter]

X = df[:-1].to_numpy()
X_cont = df[cont_attr].to_numpy()
X_sel = df[sel_attr].to_numpy()
y = df[y_attr].to_numpy()
attr = list(df[:-1].columns)

# define class labels based on target
# 0 - low chance of getting heart attack
# 1 - high chance of getting heart attack

classNames = ["Less Chance for Heart Attack", "More Chance for Heart Attack"]
classDict = dict(zip(classNames, range(len(classNames))))

# no. of objects and attributes
N, M = X.shape
N_cont, M_cont = X_cont.shape
N_sel, M_sel = X_sel.shape
# no of classes
C = len(classNames)
