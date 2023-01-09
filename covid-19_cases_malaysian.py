# %% Import packages
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras import Sequential, Input
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import os

#from module import lstm_model_creation

# %%
#Data Loading
CSV_PATH = os.path.join(os.getcwd(),'dataset','cases_malaysia_train.csv')
df = pd.read_csv(CSV_PATH)

# %%
#2. Data Inspection/ Data Visualization
df.head(10)       # To show n rows of data,
df.tail()         # To show n rows of data
df.info() 

# %%
df.describe().T

# %%
#Plot graph before remove NaN/ change datatype
plt.figure()
plt.plot(df['cases_new'].values)
plt.show()

# %%
# Convert object into numeric using to_numeric
df['cases_new'] = pd.to_numeric(df['cases_new'], errors = 'coerce')

df.info()

# %%
# check NaN --> 12
df.isna().sum()

# %%
#Removing NaN using Interpolation
df['cases_new'] = df['cases_new'].interpolate(method = 'polynomial', order = 2)

print(df.isna().sum())
print(df.info())

#Plot graph
plt.figure(figsize=(10,10))
plt.plot(df['cases_new'].values)
plt.show()

# %%
#4. Features Selection
#name the 'cases_new' to the cn
cn = df['cases_new'].values

# %%
#5. Data Preprocessing
mms = MinMaxScaler()

# Method 2) Pythonic 
#cn = mms.fit_transform(cn.reshape(-1, 1))
cn = mms.fit_transform(cn[::, None])

# %%
# Data Splitting
X = []
y = []
win_size = 30

for i in range(win_size, len(cn)):
    X.append((cn[i - win_size:i]))
    y.append((cn[i]))

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True, random_state = 123)

# %%
#6. Model Development
#Sequential Functional

model = Sequential()
model.add(LSTM(64, return_sequences = True, input_shape=(X_train.shape[1:])))
#model.add(Input(shape=(30, 1)))
model.add(Dropout(0.3))
model.add(LSTM(64, return_sequences = True))
model.add(LSTM(64, return_sequences = True))
model.add(LSTM(64))
model.add(Dense(1, activation = 'relu'))
model.summary()

plot_model(model, show_shapes=True)

model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mape'])

# model = lstm_model_creation(X_train, dropout=0.3, nodes=64)

hist = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs = 12, batch_size = 64)

# %%
# Tensorboard callback
LOGS_PATH = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime('%Y%m%d - %H%M%S'))

ts_callback = TensorBoard(log_dir = LOGS_PATH)
es_callback = EarlyStopping(monitor = 'val_los', patience = 5, verbose = 0, restore_best_weights = True)

#Train model
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs = 10, batch_size = 64, callbacks = [es_callback, ts_callback])

# %%
#7. Model Analysis/ Model Evaluation

TEST_CSV_PATH = os.path.join(os.getcwd(), 'dataset', 'cases_malaysia_test.csv')

test_df = pd.read_csv(TEST_CSV_PATH)

# %%
# check NaN --> 1
test_df.isna().sum()

# %%
#To drop NaN
test_df['cases_new'] = test_df['cases_new'].interpolate(method = 'polynomial', order = 2)
#To double check 
test_df.isna().sum()

# %%
# to concatenate the data
concat = pd.concat((df['cases_new'], test_df['cases_new']))
concat = concat[len(df['cases_new']) - win_size:]

# min max transformation
concat = mms.transform(concat[::, None])
#concat = mms.fit_transform(cn.reshape(-1, 1)) 

X_test = []
y_test = []

for i in range(win_size, len(concat)):
    X_test.append(concat[i - win_size:i])
    y_test.append(concat[i])

X_test = np.array(X_test)
y_test = np.array(y_test)

# To predict the unseen dataset
predicted = model.predict(X_test) 

# %%
# Visualize the actual and predicted
plt.figure()

plt.plot(predicted, color = 'blue')
plt.plot(y_test, color = 'red')
plt.legend(['Predicted Casses',  'Actual Casses'])
plt.xlabel('Time')
plt.ylabel('Number of cases')
plt.show()

# %%
# actual, predicted: print mape, mae and mse
print("mean absolute percentage error = ")
print(mean_absolute_percentage_error(y_test, predicted))
print("mean absolute error = ")
print(mean_absolute_error(y_test, predicted))
print("mean squared error = ")
print(mean_squared_error(y_test, predicted))

# %%
# To save train model
model.save('model.h5')

# %%
