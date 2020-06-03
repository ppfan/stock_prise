import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
split_ratio = 0.8
df = pd.read_csv('AAPL.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_axis(df['Date'], inplace=True)

close_data = df['Close'].values
close_data = close_data.reshape((-1,1))

split = int(split_ratio*len(close_data))

close_train = close_data[:split]
close_test = close_data[split:]

date_train = df['Date'][:split]
date_test = df['Date'][split:]

print(len(close_train))
print(len(close_test))

look_back = 10

train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)
test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)

model = Sequential()
model.add(
    LSTM(10,
        activation='relu',
        input_shape=(look_back,1))
)
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

num_epochs = 25
model.fit_generator(train_generator, epochs=num_epochs, verbose=1)

prediction = model.predict_generator(test_generator)

close_train = close_train.reshape((-1))
close_test = close_test.reshape((-1))
prediction = prediction.reshape((-1))

plt.plot(date_train,close_train,color = 'blue',label = 'тренировочные данные')
plt.plot(date_test,close_test,color ='g',label ='тестовые')
plt.plot(date_test[0:(252-look_back)],prediction,color = 'r',label = 'предсказание')
plt.title('Предсказание стоимости акций Apple за {} недель'.format(look_back))
plt.legend()
plt.show()
