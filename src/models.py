# models.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Bidirectional, Dense, Dropout
'''
Each of the model architecture parameters are defined below
'''
vocabulary_size = 10000
embedded_dimensions = 100
hidden_size = 64
dropout_size = 0.4

''' 
Each function builds the RNN model based on the sequence length associated with it. These models are called in the train.py file for each of the configurations. 

Each model consists of an embedding layer, 2 hidden layers and an output layer. The models which have been built here are Recurrent Neural networks, Long Short Term Memory networks, and Bidirectional Long Short Term memory networks
'''

def build_rnn(seq_length, activation="relu"):
    model = Sequential([Embedding(vocabulary_size, embedded_dimensions, input_length=seq_length),SimpleRNN(hidden_size, return_sequences=False),Dropout(dropout_size),Dense(hidden_size, activation=activation),Dropout(dropout_size),Dense(1, activation="sigmoid")])
    return model

def build_lstm(seq_length, activation="relu"):
    model = Sequential([Embedding(vocabulary_size, embedded_dimensions, input_length=seq_length),LSTM(hidden_size, return_sequences=False),Dropout(dropout_size),Dense(hidden_size, activation=activation),Dropout(dropout_size),Dense(1, activation="sigmoid")])
    return model

def build_bilstm(seq_length, activation="relu"):
    model = Sequential([Embedding(vocabulary_size, embedded_dimensions, input_length=seq_length),Bidirectional(LSTM(hidden_size, return_sequences=False)),Dropout(dropout_size),Dense(hidden_size, activation=activation),Dropout(dropout_size),Dense(1, activation="sigmoid")])
    return model
