
import pandas as pd
import numpy
import matplotlib.pyplot as plt
#from pandas import read_csv
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model

dataframe = pd.read_csv('analysisData/California-Alameda.csv', usecols=[3])
dataset = dataframe.values

# 将整型变为float
dataset = dataset.astype('float32')
print(dataset)
plt.plot(dataset)
plt.show()

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY),dataX

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

look_back = 1
testX, testY ,tdata= create_dataset(test, look_back)
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = load_model('model/newmodel.h5',compile = False)

testPredict = model.predict(testX)

testPredict = scaler.inverse_transform(testPredict)
testY=scaler.inverse_transform([testY])

testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[look_back:len(testPredict)+look_back, :] = testPredict

plt.plot(scaler.inverse_transform(dataset))
plt.plot(testPredictPlot)
plt.show()
