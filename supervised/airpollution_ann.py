# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("bangkok-air-quality.csv")

df = df.replace(' ', np.nan)
df = df.dropna(subset=[' pm25', ' pm10', 'date'])

df

# %%
df[[' pm25', ' pm10', ' o3', ' no2', ' so2', ' co']] = df[[' pm25', ' pm10', ' o3', ' no2', ' so2', ' co']].astype(float)
df[[' pm25', ' pm10', ' o3', ' no2', ' so2', ' co']] = df[[' pm25', ' pm10', ' o3', ' no2', ' so2', ' co']].astype(pd.Int32Dtype())

df['date'] = pd.to_datetime(df['date'])

df['dayofweek'] = df['date'].dt.dayofweek
df['week'] = df['date'].dt.week
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

df['date'] = df['date'].astype(int)

df['date'] = (df['date'] - df['date'].min()) / (df['date'].max() - df['date'].min())
df['year'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())

df = df.sort_values(by='date')
df = df.reset_index(drop=True)

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)

df = df.sort_index()

# encode
df_day_of_week = pd.DataFrame(encoder.fit_transform(df['dayofweek'].values.reshape(-1, 1)))
df_week = pd.DataFrame(encoder.fit_transform(df['week'].values.reshape(-1, 1)))
df_day = pd.DataFrame(encoder.fit_transform(df['day'].values.reshape(-1, 1)))
df_month = pd.DataFrame(encoder.fit_transform(df['month'].values.reshape(-1, 1)))

# index
df_day_of_week.index = df.index
df_week.index = df.index
df_day.index = df.index
df_month.index = df.index



df = df.drop(['dayofweek'], 1)
df = df.drop(['week'], 1)
df = df.drop(['day'], 1)
df = df.drop(['month'], 1)

df = df.drop([' o3'], axis=1)
df = df.drop([' no2'], axis=1)
df = df.drop([' so2'], axis=1)
df = df.drop([' co'], axis=1)


df = pd.concat([df_day_of_week, df], axis=1)

df

# %%
X, y = df.drop([' pm25', ' pm10'], axis=1), df[[' pm25', ' pm10']]

x_train, x_test = X[:int(len(X)*0.9)], X[int(len(X)*0.9):]
y_train, y_test = y[:int(len(y)*0.9)], y[int(len(y)*0.9):]

print(x_train), print(y_train)

# %%
from sklearn.neural_network import MLPRegressor

ANN = MLPRegressor(activation='relu', solver='adam', hidden_layer_sizes=(64, 64), n_iter_no_change=200, learning_rate='constant', max_iter=600000, random_state=42, verbose=True, max_fun=20000, tol=0.05)

ANN.fit(x_train, y_train)

# %%
prediction = ANN.predict(x_train)
prediction = np.array(prediction)


plt.scatter([i for i in range(len(prediction[:,0]))], y_train.values[:,0], color='royalblue')
plt.plot([i for i in range(len(prediction[:,0]))], prediction[:,0], color='mediumseagreen')

# %%
prediction = ANN.predict(x_test)
prediction = np.array(prediction)


plt.scatter([i for i in range(len(prediction[:,0]))], y_test.values[:,0], color='royalblue')
plt.plot([i for i in range(len(prediction[:,0]))], prediction[:,0], color='mediumseagreen')

plt.scatter([i for i in range(len(prediction[:,0]))], y_test.values[:,1], color='hotpink')
plt.plot([i for i in range(len(prediction[:,0]))], prediction[:,1], color='khaki')

plt.title('PM 2.5 & PM10 Prediction using RNN')
plt.show()

# %%



