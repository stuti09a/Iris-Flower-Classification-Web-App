# data_exploration.py
# Simple script to output basic univariate stats and save histogram images
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt

iris = load_iris(as_frame=True)
df = iris.frame
print(df.describe())

for col in df.columns:
    plt.figure()
    df[col].hist(bins=20)
    plt.title(col)
    plt.savefig(f"{col}_hist.png")
print("Saved histogram PNGs")
