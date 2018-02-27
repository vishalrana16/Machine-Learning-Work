import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset_train = pd.read_csv('prices.csv')

yahoo = dataset_train[dataset_train['symbol'] == 'YHOO'].reset_index()

yahoo.drop('index', inplace = True, axis = 1)   

yahoo_data = yahoo.iloc[:,3:4].values

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))
yahoo_data_scaled = sc.fit_transform(yahoo_data)

X_train = []
y_train = []

for i in range(36, 1562):
    X_train.append(yahoo_data_scaled[i-36:i,0])
    y_train.append(yahoo_data_scaled[i,0])

X_test =[]
y_test = []
    
for i in range(1562, 1762):
    X_test.append(yahoo_data_scaled[i-36:i,0])
    y_test.append(yahoo_data_scaled[i,0])


X_train = np.array(X_train)    
y_train = np.array(y_train)  

X_test = np.array(X_test)
y_test = np.array(y_test)

X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1],1))
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


model = Sequential()
model.add(LSTM(units = 64, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 64, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 128, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 128))
model.add(Dropout(0.2))
model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(X_train, y_train, epochs = 100, batch_size = 32)

model.save('model1.h5')

y_pred = model.predict(X_test)
y_pred = y_pred.reshape(200,)

y_pred = sc.inverse_transform(y_pred)

y_test = sc.inverse_transform(y_test)


plt.plot(y_test, color = 'red', label = 'Real Yahoo Stock Price(close)')
plt.plot(y_pred, color = 'blue', label = 'Predicted Yahoo Stock Price(close)')
plt.title('Yahoo Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Yahoo Stock Price')
plt.legend()
plt.show()


  