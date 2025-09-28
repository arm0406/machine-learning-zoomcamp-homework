import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv")
df.head()
df.head(15)
df.shape
df.describe()
df['fuel_type'].value_counts()
pd.__version__
null_values=df.columns[df.isna().any()]
len(null_values)
df['origin'].value_counts()
