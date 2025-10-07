import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv")
pd.__version__
df.shape
df['fuel_type'].value_counts()
null_values=df.columns[df.isna().any()]
len(null_values)
df.groupby('origin')['fuel_efficiency_mpg'].max()['Asia']
df['horsepower'].median()
df['horsepower'].mode()[0]
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].mode()[0])
df['horsepower'].median()
asia_cars = df[df['origin'] == 'Asia']
cols = ['vehicle_weight', 'model_year']
selected_asia_cars =asia_cars[cols].head(7)
X = selected_asia_cars.to_numpy()
XTX = X.T.dot(X)
XTX_inv = np.linalg.inv(XTX)
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
w = XTX_inv.dot(X.T).dot(y)
result = w.sum()
print(result)
