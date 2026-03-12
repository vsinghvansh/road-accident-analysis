# road-accident-analysis
This project analyzes road accident casualty data to identify patterns in accident severity, age groups, and gender distribution. Using Python libraries such as Pandas, NumPy, Matplotlib, and Seaborn, exploratory data analysis and visualizations were performed to uncover key trends and insights related to road safety.

import warnings 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Updated Local File Path
df = pd.read_csv('/Users/vanshsingh/Downloads/dft-road-casualty-statistics-casualty-provisional-mid-year-unvalidated-2023.csv')

df.head()
df.info()
df.isna().sum()
df.describe()

df[df.columns[1]].dtype == 'int64'

for col in df.columns:
    if df[col].dtype == 'object':
        print(col)

for col in df.columns:
    print(f'{col} : {df[col].value_counts().size}')

plt.figure(figsize=(8, 6))
sns.countplot(x='casualty_severity', data=df)
plt.title('Distribution of Accident Severities')
plt.xlabel('Casualty Severity')
plt.ylabel('Count')
plt.show()

sns.boxplot(x='casualty_severity', y='age_of_casualty', data=df)

plt.figure(figsize=(8, 6))
sns.countplot(x='casualty_severity', hue='sex_of_casualty', data=df)
plt.legend(title='Sex of Casualty', labels=['Unknown', 'Male', 'Female'])
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='casualty_severity', hue='casualty_class', data=df)
plt.title('Casualty Severity by Casualty Class')
plt.xlabel('Casualty Severity')
plt.ylabel('Count')
plt.legend(title='Casualty Class', labels=['Driver', 'Passenger', 'Pedestrian'])
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='casualty_severity', hue='casualty_home_area_type', data=df)
plt.title('Casualty Severity by Home Area Type')
plt.xlabel('Casualty Severity')
plt.ylabel('Count')
plt.legend(title='Home Area Type', labels=['Unknown', 'Urban', 'Semi-Urban', 'Rural'])
plt.show()

sns.boxplot(x='casualty_severity', y='casualty_imd_decile', data=df)

sns.countplot(x='casualty_type', data=df)

sns.boxplot(x='casualty_severity', y='casualty_type', data=df)

corr = df.corr(numeric_only=True)

mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(15, 12))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.title('Correlation Matrix Heatmap')
plt.show()
