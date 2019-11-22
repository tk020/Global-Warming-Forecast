import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def process_FAA_hourly_data(filename):
    path = os.getcwd()
    pathfile = os.path.join(path,"..","data",filename)
    df_temps = pd.read_csv(pathfile, skiprows=16)
    df_temps = df_temps.iloc[:,:-1]
    df_temps = df_temps.loc[df_temps[df_temps.columns[0]] != df_temps.columns[0]]
    df_temps[df_temps.columns[1]] = df_temps[df_temps.columns[1]].apply(pd.to_numeric, downcast = "integer")
    df_temps[df_temps.columns[2:]] = df_temps[df_temps.columns[2:]].apply(pd.to_numeric, downcast = "float")
    df_temps = df_temps.set_index(pd.DatetimeIndex(df_temps[df_temps.columns[0]]))
    df_temps = df_temps.drop([df_temps.columns[0]], axis=1)
    return df_temps

df_kphl = process_FAA_hourly_data("faa_hourly-KPHL_20120101-20190101.csv")
df_kphl_useful = df_kphl.iloc[:,[1,7,8,9]]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

df_kphl_useful[[df_kphl_useful.columns[0],df_kphl_useful.columns[1],df_kphl_useful.columns[2],df_kphl_useful.columns[3]]] = scaler.fit_transform(df_kphl_useful[[df_kphl_useful.columns[0],df_kphl_useful.columns[1],df_kphl_useful.columns[2],df_kphl_useful.columns[3]]])

start = df_kphl_useful.index[0]
end = df_kphl_useful.index[-1]
idx = pd.date_range(start, end, freq='H')

df_kphl_useful = df_kphl_useful.reindex(idx, fill_value = np.nan)

print("Start day is {}".format(df_kphl_useful.index[0]))
print("End day is {}".format(df_kphl_useful.index[-1]))

#Stacked auto encoder
from keras.layers import *
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, RepeatVector

dataset = np.array(df_kphl_useful)

dataset = np.nan_to_num(dataset)
train, test = dataset[0:50,:], dataset[(1713*35):len(dataset),:]
print(len(train), len(test))


#Get Only temperature data
def create_datasettemp(dataset, look_back=25):
    data = []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0].flatten()
        data.append(a)
    return np.array(data)

#Get all four variables
def create_dataset(dataset, look_back=24):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :].flatten()
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


#Tt=f(Tt−1,Tt−2,…..Tt−n,Pt−1,Pt−2,…..Pt−n,Ht−1,Ht−2,…..Ht−n,Wt−1,Wt−3,…..Wt−n)(3)

#train X will have 41087 sets of windows and each window has 96 variables (24*4)
#train Y will have 41087 sets of windoes and one variable for each time (temperature)

# reshape into X=t and Y=t+1
look_back = 10
traindata = create_datasettemp(train, look_back)
testdata = create_datasettemp(test, look_back)


traindata.shape

trainX = traindata[:,0:-1]
trainY = traindata[:,-1]

testX = testdata[:,0:-1]
testY = testdata[:,-1]

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print(trainX.shape, testX.shape, trainY.shape, testY.shape)

model = Sequential()
model.add(LSTM(24, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(trainX, trainY, epochs=50, batch_size=72, verbose=2, shuffle=False)
