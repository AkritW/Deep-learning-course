# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import this

df = pd.read_csv("bangkok-air-quality.csv")

df = df.replace(' ', np.nan)
df = df.dropna(subset=[' pm25', ' pm10', 'date'])

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


df_year = df['year']
df = df.drop(['year'], axis=1)
# df = pd.concat([df_day_of_week, df_week, df_day, df_month, year, df], axis=1)

# del #
df = pd.concat([df], axis=1)
########




# from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(df.drop([' pm25', ' pm10'], axis=1), df[[' pm25', ' pm10']], test_size=0.3, random_state=42) 


# x_train = x_train.sort_index()
# x_test = x_test.sort_index()
# y_train = y_train.sort_index()
# y_test = y_test.sort_index()

df

# %%
import pandas as pd
from collections import deque
import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization, Flatten
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, ModelCheckpoint
import time
from sklearn import preprocessing
from collections import deque
from sklearn.model_selection import train_test_split

# %%
# preprocessing

SEQ_LEN = 90

sequential_data = []
prev_days = deque(maxlen=SEQ_LEN)

for i in df.values:
    prev_days.append([n for n in i[:-2]])
    if len(prev_days) == SEQ_LEN:
        sequential_data.append([np.array(prev_days), i[-2:]])

X = []
y = []
for x_val, y_val in sequential_data:
    X.append(x_val)
    y.append(y_val)

x_train, y_train = X[:int(len(X)*0.9)], y[:int(len(y)*0.9)]
x_train, y_train = np.array(x_train).astype('float32'), np.array(y_train).astype('float32')

x_test, y_test= X[int(len(X)*0.9):], y[int(len(y)*0.9):]
x_test, y_test = np.array(x_test).astype('float32'), np.array(y_test).astype('float32')


x_train.shape, y_train.shape, x_test.shape, y_test.shape

# %%
model = Sequential()

LSTM_RETURN_LAYER = 2
LSTM_RETURN_SIZE = 300
LSTM_NORMAL_LAYER = 1
LSTM_NORMAL_SIZE = 300
DENSE_LAYER = 0
DENSE_SIZE = 60
DROPOUT_SIZE = 0.3
BATCH_SIZE = 24
EPOCH = 960

for i in range(LSTM_RETURN_LAYER):
    model.add(LSTM(LSTM_RETURN_SIZE, input_shape=(x_train.shape[1:]), return_sequences=True))
    model.add(Dropout(DROPOUT_SIZE))
    model.add(BatchNormalization())

for i in range(LSTM_NORMAL_LAYER):
    model.add(LSTM(LSTM_NORMAL_SIZE))
    model.add(BatchNormalization())


for i in range(DENSE_LAYER):
    model.add(Dense(DENSE_SIZE, activation='relu'))

# output layer
model.add(Dense(2))

opt = tf.keras.optimizers.Adam(learning_rate=0.01, decay=0.001)

model.compile(
    loss='mse',
    optimizer=opt
)

name1 = f'WeatherForcast-{LSTM_RETURN_LAYER}-LSTM_RETURN_LAYER-{LSTM_RETURN_SIZE}-LSTM_RETURN_SIZE'
name2 = f'-{LSTM_NORMAL_LAYER}-LTSM_NORMAL_LAYER-{LSTM_NORMAL_SIZE}-LSTM_NORMAL_SIZE-{DENSE_LAYER}-DENSE_LAYER'
name3 = f'-{DENSE_SIZE}-DENSE_SIZE-{DROPOUT_SIZE}-DROPOUT_SIZE-{BATCH_SIZE}-BATCH_SIZE-{EPOCH}-EPOCH'
NAME  = name1 + name2 + name3

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))



history = model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCH,
    validation_data=(x_test, y_test), 
    verbose=2,
    shuffle=True,
)

# %%
# Score model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score)

# %%
prediction = model.predict(x_train)
prediction = np.array(prediction)


plt.scatter([i for i in range(len(prediction[:,0]))], y_train[:,0], color='royalblue')
plt.plot([i for i in range(len(prediction[:,0]))], prediction[:,0], color='mediumseagreen')

plt.scatter([i for i in range(len(prediction[:,0]))], y_train[:,1], color='hotpink')
plt.plot([i for i in range(len(prediction[:,0]))], prediction[:,1], color='khaki')

plt.title('PM 2.5 & PM10 Prediction using RNN')
plt.show()

# %%
prediction = model.predict(x_test)
prediction = np.array(prediction)


plt.scatter([i for i in range(len(prediction[:,0]))], y_test[:,0], color='royalblue')
plt.plot([i for i in range(len(prediction[:,0]))], prediction[:,0], color='mediumseagreen')

plt.scatter([i for i in range(len(prediction[:,0]))], y_test[:,1], color='hotpink')
plt.plot([i for i in range(len(prediction[:,0]))], prediction[:,1], color='khaki')

plt.title('PM 2.5 & PM10 Prediction using RNN')
plt.show()

# %%



