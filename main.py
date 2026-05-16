# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("data/StudentsPerformance.csv")

# Display Dataset
print("\nFirst 5 Rows:")
print(df.head())

# Dataset Information
print("\nDataset Info:")
print(df.info())

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate Values
print("\nDuplicate Rows:", df.duplicated().sum())

# -----------------------------
# Data Visualization
# -----------------------------

sns.set(style="whitegrid")

# Histogram
plt.figure(figsize=(8,5))
sns.histplot(df['math score'], bins=20, kde=True)
plt.title("Distribution of Math Scores")
plt.xlabel("Math Score")
plt.ylabel("Frequency")
plt.savefig("images/math_score_distribution.png")
plt.show()

# Boxplot
plt.figure(figsize=(8,5))
sns.boxplot(x='gender', y='math score', data=df)
plt.title("Gender vs Math Score")
plt.savefig("images/gender_math_boxplot.png")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
correlation = df[['math score', 'reading score', 'writing score']].corr()

sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("images/correlation_heatmap.png")
plt.show()

# Pairplot
sns.pairplot(df[['math score', 'reading score', 'writing score']])
plt.savefig("images/pairplot.png")
plt.show()

# Average Scores by Test Preparation
avg_scores = df.groupby('test preparation course')[
    ['math score', 'reading score', 'writing score']
].mean()

print("\nAverage Scores by Test Preparation:")
print(avg_scores)

# Bar Plot
avg_scores.plot(kind='bar', figsize=(8,5))
plt.title("Average Scores by Test Preparation")
plt.ylabel("Scores")
plt.xticks(rotation=0)
plt.savefig("images/test_preparation_scores.png")
plt.show()

# Insights
print("\nKey Insights:")
print("1. Students completing test preparation scored higher.")
print("2. Reading and writing scores are strongly correlated.")
print("3. Female students performed better in reading and writing.")
