import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("heart.csv")
cols = np.asarray(df.columns)

data = df.to_numpy()

print(cols)
print(data)

thal = np.transpose(data)[12]

plt.hist(thal, bins=4, rwidth=0.75, color=['red', 'blue', 'green', 'yellow'])
plt.show()
