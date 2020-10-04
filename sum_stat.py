import pandas as pd
import numpy as np

df = pd.read_csv("heart.csv")
cols = np.asarray(df.columns)
rows = ["mean", "median", "std. dev.", "variance"]

data = df.to_numpy()

mean_val = np.mean(data, 0)
med_val = np.median(data, 0)
std_val = np.std(data, 0)
var_val = np.var(data, 0)

out = pd.DataFrame(np.transpose(np.array([mean_val, med_val, std_val, var_val])),
                   columns=rows)

print(out.to_latex(float_format="%.2f"))