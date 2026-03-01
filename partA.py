import pandas as pd
import matplotlib.pyplot as plt

path = "/Users/charlotte/Downloads/concrete+compressive+strength/Concrete_Data.xls"

#read data
data = pd.read_excel(path, engine='xlrd')
# print(data.columns)
data.columns = [
    "Cement", "Blast_Furnace_Slag", "Fly_Ash", "Water",
    "Superplasticizer", "Coarse_Aggregate", "Fine_Aggregate",
    "Age", "Concrete_Strength"
]

#statistics
summary_stats = data.describe()
print(summary_stats)

#correlation matrix
correlation_matrix = data.corr()
print("Correlation Matrix:")
print(correlation_matrix)

#strength: histogram
plt.hist(data["Concrete_Strength"], bins=20, edgecolor='black')
plt.title("Distribution of Concrete Strength")
plt.xlabel("Concrete Strength (MPa)")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.75)
plt.show()

#cement vs strength: scatter plot
plt.scatter(data["Cement"], data["Concrete_Strength"], alpha=0.5)
plt.title("Cement vs Concrete Strength")
plt.xlabel("Cement (kg/m^3)")
plt.ylabel("Concrete Strength (MPa)")
plt.grid()
plt.show()

#age vs strength: scatter plot
plt.scatter(data["Age"], data["Concrete_Strength"], alpha=0.5)
plt.title("Age vs Concrete Strength")
plt.xlabel("Age (days)")
plt.ylabel("Concrete Strength (MPa)")
plt.grid()
plt.show()


