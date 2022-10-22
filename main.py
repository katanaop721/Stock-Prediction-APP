import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import pandas_datareader as data # scraping data
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

start = '2010-01-1'
end = '2019-12-31'

df = data.DataReader('AAPL', 'yahoo', start, end)
df = df.reset_index() # for starting index from 0 1 2 3 not from date
df = df.drop(['Date', 'Adj Close'], axis=1) # droping two colums

# showing liquid line in graph

ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

plt.figure(figsize=(12,6))
plt.plot(df.Close)

plt.plot(ma100, 'r')
plt.plot(ma200, 'g')

plt.show()

# spliting data into training

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)]) # taking 70 % data for training
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))]) # taking 30 % data for testing

# print(data_training.shape)
# print(data_testing.shape)

scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

# now
# 33 36 33 40 39 38 37 42 44 38 # 10 day data Feture
# 11th data depend on uper 10 days data Result (Predicted Data)

x_train = []
y_train = []

# data_training_array.shape[0] =  num of data = (1761, 1)

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

# ML MODEL



'''
    The Sequential model is a linear stack of layers. The common architecture of ConvNets is a sequential architecture. 
However, some architectures are not linear stacks. For example, siamese networks are two parallel neural networks with 
some shared layers. More examples here.
'''

model = Sequential()
model.add(LSTM(units= 50, activation = 'relu', return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))

# model = Sequential()
model.add(LSTM(units= 60, activation = 'relu', return_sequences = True))
model.add(Dropout(0.3))

# model = Sequential()
model.add(LSTM(units= 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.4))

# model = Sequential()
model.add(LSTM(units= 120, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))

# model.compile(optimizer='adam', loss='mean_squared_error')
# model.fit(x_train, y_train, epochs=50)

# model.save('keras_model.h5')

# model.summary()

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scale_factor = 1/0.02099517
y_predicted = y_predicted * scale_factor

y_test = y_test*scale_factor

# print(y_predicted)
plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Logestic Regression

logistic_model = LogisticRegression()
print(x_train[0],y_train)
logistic_model.fit(x_train, y_train)
y_pred = logistic_model.predict(x_test)
print(accuracy_score(y_pred,y_test))