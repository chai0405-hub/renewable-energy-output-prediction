import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("data/energy_data.csv")

X = df[['temperature', 'wind_speed', 'solar_irradiance']]
y = df['energy_output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

pickle.dump(model, open("model/trained_model.pkl", "wb"))
print("Model trained and saved")