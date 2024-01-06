import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.constraints import max_norm
import tensorflow as tf
import streamlit as st
import geopandas as gpd
from shapely.geometry import Point, Polygon
import folium
from geopy.geocoders import Nominatim as NT  
# from geopandas.tools import geopandas_geoseries_geometries

speed_data = pd.read_csv("los_speed.csv")
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

# Load the model architecture
model = Sequential()
model.add(SimpleRNN(units=32, activation='relu', input_shape=(207, 10), return_sequences=True, kernel_constraint=max_norm(3)))
model.add(LSTM(units=64, activation='relu'))
model.add(Dense(units=207, activation='linear'))
model.add(Dropout(0.2))
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=3e-4), loss='mean_squared_error')

# Load the saved weights
weights_filename = "model_weights.h5"
model.load_weights(weights_filename)

predictions = model.predict(testX)
pred_rescref = np.array(predictions * max_speed)
print("pred", pred_rescref)

# Set up your Streamlit app
st.set_page_config(page_title="Traffic Prediction", page_icon=":smiley:")
# st.header("Traffic Prediction")
st.title("Traffic Prediction")
st.write("This is a web app to predict the traffic in the Netherlands")

# Take input from the user for time from 0 to 100 in multiples of 5
time = st.slider("Time (In minutes)", 0, 300, 25, 5)
st.write("Time is", time, "minutes")

def plotter(time):

    lat = []
    log = []
    coord = []
    for j in range(207):
        print(abs(pred_rescref[time//5][j]))
        if pred_rescref[time//5][j] <= 20:
            lat.append(latitude[j])
            log.append(longitude[j])
            coord.append({latitude[j], longitude[j]})

    data = {'Latitude': lat, 'Longitude': log}
    df = pd.DataFrame(data)

    if len(lat) == 0:
        st.warning("No points to plot.")
        return
    else :
        map_center = [lat[0], log[0]]
        # centerx, centery = 0, 0
        # for i in lat:
        #     centerx += i
        # centerx = centerx/len(lat)
        # for j in log:
        #     centery += j
        # centery = centery/len(log)
        # map_center = [centerx, centery]
    
    mymap = folium.Map(location=map_center, zoom_start=12)

    for lat_, lon_ in zip(lat, log):
        folium.CircleMarker(location=[lat_, lon_], radius=5, color='purple', fill=True, fill_color='purple', fill_opacity=0.9, popup=f"Coordinates: ({lat_}, {lon_})").add_to(mymap)
    
    # st.title("Google Map with Marked Points")
    # st.write("This is a Streamlit app with a Google Map.")

    # Convert folium map to HTML and display using st.components.v1.html
    map_html = folium.Map(location=map_center, zoom_start=12)._repr_html_()
    st.components.v1.html(map_html, width=800, height=600)

    print(data)

plotter(time)


