import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from keras import Sequential
from keras.layers import Dense,SimpleRNN,Embedding,Flatten, LSTM, GlobalAveragePooling3D
from keras.optimizers import Adam
from keras.constraints import max_norm
from keras.layers import Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

speed_data = pd.read_csv('los_speed.csv')
speed_data = speed_data.transpose()
print(type(speed_data))

num_nodes, time_len = speed_data.shape
print(num_nodes, time_len)

nan_count = np.isnan(speed_data).sum()
print(nan_count)

start_value_latitude = 52.3000
start_value_longitude = 4.8800
increment = 0.0004
num_values = 207
latitude = [round(start_value_latitude + i * increment, 4) for i in range(num_values)]
longitude = [round(start_value_longitude + i * increment, 4) for i in range(num_values)]

def train_test_split(data, train_portion):
    time_len = data.shape[1]
    train_size = int(time_len * train_portion)
    train_data = np.array(data.iloc[:, :train_size])
    test_data = np.array(data.iloc[:, train_size:])
    return train_data, test_data

train_data, test_data = train_test_split(speed_data, 0.8)
print("Train data: ", train_data.shape)
print("Test data: ", test_data.shape)

max_speed = train_data.max()

def scale_data(train_data, test_data):
    max_speed = train_data.max()
    min_speed = train_data.min()
    if(max_speed == min_speed):
        train_scaled = train_data / max_speed
        test_scaled = test_data / max_speed
    else:
        train_scaled = (train_data - min_speed) / (max_speed - min_speed)
        test_scaled = (test_data - min_speed) / (max_speed - min_speed)
    return train_scaled, test_scaled

train_scaled, test_scaled = scale_data(train_data, test_data)

seq_len = 10
pre_len = 12

def sequence_data_preparation(seq_len, pre_len, train_data, test_data):
    trainX, trainY, testX, testY = [], [], [], []

    for i in range(train_data.shape[1] - int(seq_len + pre_len - 1)):
        a = train_data[:, i : i + seq_len + pre_len]
        trainX.append(a[:, :seq_len])
        trainY.append(a[:, -1])

    for i in range(test_data.shape[1] - int(seq_len + pre_len - 1)):
        b = test_data[:, i : i + seq_len + pre_len]
        testX.append(b[:, :seq_len])
        testY.append(b[:, -1])

    trainX = np.array(trainX)
    trainY = np.array(trainY)
    testX = np.array(testX)
    testY = np.array(testY)

    return trainX, trainY, testX, testY

trainX, trainY, testX, testY = sequence_data_preparation(
    seq_len, pre_len, train_scaled, test_scaled
)

print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)

res = []

model = Sequential()
model.add(SimpleRNN(units=32, activation='relu', input_shape=(207, 10), return_sequences=True, kernel_constraint=max_norm(3)))
# model.add(SimpleRNN(units=32, activation='relu', return_sequences=True, kernel_constraint=max_norm(3)))
# model.add(SimpleRNN(units=32, activation='relu', return_sequences=True, kernel_constraint=max_norm(3)))
# model.add(LSTM(units=64, activation='relu', return_sequences=True))
# model.add(LSTM(units=64, activation='relu', return_sequences=True))
model.add(LSTM(units=64, activation='relu'))
model.add(Dense(units=207, activation='relu'))
# model.add(LSTM(units=64, activation='relu'))
# model.add(Dense(units=207, activation='relu'))
# model.add(LSTM(units=64, activation='relu'))
# model.add(Dense(units=207, activation='relu'))
model.add(Dropout(0.2))
model.compile(optimizer=Adam(learning_rate=3e-4), loss='mean_squared_error')
history = model.fit(trainX, trainY, epochs=25, batch_size=32, validation_split=0.2)

weights_filename = "model_weights.h5"
model.save_weights(weights_filename)

predictions = model.predict(testX)
pred_rescref = np.array(predictions * max_speed)
test_rescref = np.array(testY * max_speed)
print(pred_rescref.shape)
rmse = np.sqrt(mean_squared_error(testY, predictions))

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(testY, predictions)
print("Mean Absolute Percentage Error:", mape)

#res.append({i, rmse})
    # print(i, rmse)

print("Root Mean Squared Error", rmse)
