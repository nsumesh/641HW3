# models.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Bidirectional, Dense, Dropout

vocabulary_size = 10000
embedded_dimensions = 100
hidden_size = 64
droupout_size = 0.4

def build_rnn(seq_length, activation="relu"):
    model = Sequential([
        Embedding(vocabulary_size, embedded_dimensions, input_length=seq_length),
        SimpleRNN(hidden_size, return_sequences=False),
        Dropout(droupout_size),
        Dense(hidden_size, activation=activation),
        Dropout(droupout_size),
        Dense(1, activation="sigmoid")
    ])
    return model

def build_lstm(seq_length, activation="relu"):
    model = Sequential([
        Embedding(vocabulary_size, embedded_dimensions, input_length=seq_length),
        LSTM(hidden_size, return_sequences=False),
        Dropout(droupout_size),
        Dense(hidden_size, activation=activation),
        Dropout(droupout_size),
        Dense(1, activation="sigmoid")
    ])
    return model

def build_bilstm(seq_length, activation="relu"):
    model = Sequential([
        Embedding(vocabulary_size, embedded_dimensions, input_length=seq_length),
        Bidirectional(LSTM(hidden_size, return_sequences=False)),
        Dropout(droupout_size),
        Dense(hidden_size, activation=activation),
        Dropout(droupout_size),
        Dense(1, activation="sigmoid")
    ])
    return model
