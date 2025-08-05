import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("Titanic-Dataset.csv")

# 1. View first 5 rows
print("First 5 Rows of the Dataset:")
print(df.head())

# 2. Dataset Info
print("\nDataset Info:")
print(df.info())

# 3. Missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# 4. Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# 5. Histogram for Age
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'].dropna(), kde=True, bins=30)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("age_distribution.png")
plt.show()

# 6. Histogram for Fare
plt.figure(figsize=(8, 5))
sns.histplot(df['Fare'], kde=True, bins=30)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("fare_distribution.png")
plt.show()

# 7. Boxplot for Age
plt.figure(figsize=(8, 5))
sns.boxplot(y='Age', data=df)
plt.title("Boxplot of Age")
plt.tight_layout()
plt.savefig("boxplot_age.png")
plt.show()

# 8. Boxplot for Fare
plt.figure(figsize=(8, 5))
sns.boxplot(y='Fare', data=df)
plt.title("Boxplot of Fare")
plt.tight_layout()
plt.savefig("boxplot_fare.png")
plt.show()

# 9. Correlation Heatmap (numeric data only)
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=['int64', 'float64'])  # Only numeric columns
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# 10. Pairplot for selected numeric features
selected_features = ['Survived', 'Pclass', 'Age', 'Fare', 'SibSp', 'Parch']
sns.pairplot(df[selected_features], hue='Survived')
plt.savefig("pairplot_selected.png")
plt.show()
