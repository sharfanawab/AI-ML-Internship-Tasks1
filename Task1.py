import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load Iris Dataset
df = sns.load_dataset("iris")

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nStatistics:")
print(df.describe())

# Scatter Plot
plt.figure(figsize=(8,5))
sns.scatterplot(
    data=df,
    x="sepal_length",
    y="petal_length",
    hue="species"
)
plt.title("Scatter Plot")
plt.savefig("scatter_plot.png")
plt.close()

# Histogram
df.hist(figsize=(10,8))
plt.suptitle("Histograms")
plt.savefig("histogram.png")
plt.close()

# Box Plot
plt.figure(figsize=(8,5))
sns.boxplot(data=df)
plt.title("Box Plot")
plt.savefig("boxplot.png")
plt.close()

print("\nFiles Saved Successfully!")

print("Current Folder:", os.getcwd())