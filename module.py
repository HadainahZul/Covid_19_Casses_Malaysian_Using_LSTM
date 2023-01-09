# %%
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Sequential
import re

# %%
def lstm_model_creation(X_train, dropout=0.3, nodes=64):
    model = Sequential()
    model.add(LSTM(64, return_sequences = True, input_shape=(X_train.shape[1:])))
    #model.add(Input(shape=(30, 1)))
    model.add(Dropout(dropout))
    model.add(LSTM(nodes, return_sequences = True))
    model.add(LSTM(nodes, return_sequences = True))
    model.add(Dropout(dropout))
    model.add(LSTM(nodes))
    model.add(Dense(1, activation = 'relu'))
    model.summary()

    plot_model(model, show_shapes=True)

    model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mape'])
    
    return model