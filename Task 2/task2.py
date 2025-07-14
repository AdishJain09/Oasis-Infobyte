import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

try:
    df = pd.read_csv("C:/Users/USER/.vscode/python/Intern/Task 2/Unemployment_Rate_upto_11_2020.csv")
except FileNotFoundError:
    print("Error: The file 'Unemployment_Rate_upto_11_2020.csv' was not found at the specified path.")
    print("Please ensure the CSV file is in 'C:/Users/USER/.vscode/python/Intern/Task 2/' or update the path in the code.")
    exit()

print("🔍 Dataset Preview:")
print(df.head())
print("\n📋 Dataset Info:")
print(df.info())

df.columns = [
    'States',
    'Date',
    'Frequency',
    'Estimated Unemployment Rate (%)',
    'Estimated Employed',
    'Estimated Labour Participation Rate (%)',
    'Region', 
    'Longitude',
    'Latitude'
]

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

print("\n❗ Missing Values:")
print(df.isnull().sum())

print("\n🗺️ Unique Regions:")
print(df['Region'].unique())
print("\n🏙️ Number of States:")
print(df['States'].nunique())


plt.figure(figsize=(16, 10))
state_avg = df.groupby('States')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False)
sns.barplot(x=state_avg.values, y=state_avg.index, palette="Reds_r")
plt.title("📊 Average Unemployment Rate by State")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("States")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 7))
sns.lineplot(data=df, x='Date', y='Estimated Unemployment Rate (%)', hue='Region', marker='o')
plt.title("📈 Unemployment Rate Over Time by Region")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
numeric_df = df.select_dtypes(include=['number'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("🧠 Correlation Heatmap of Employment Data")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Region', y='Estimated Unemployment Rate (%)', data=df, palette='Set3')
plt.xticks(rotation=45)
plt.title("📦 Unemployment Rate Distribution by Region")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 7))
sns.lineplot(x='Date', y='Estimated Labour Participation Rate (%)', hue='Region', data=df, marker='o')
plt.title("📉 Labour Participation Rate Over Time by Region")
plt.xlabel("Date")
plt.ylabel("Participation Rate (%)")
plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

df.to_csv("Cleaned_Unemployment_Data.csv", index=False)
print("✅ Cleaned dataset saved as 'Cleaned_Unemployment_Data.csv'")