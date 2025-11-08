import os
import re
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

dataset = "/Users/nsumesh/Documents/GitHub/641HW3/data/IMDB Dataset 2.csv"
output_dir = "/Users/nsumesh/Documents/GitHub/641HW3/data/preprocessed"
vocabulary_size = 10000
sequence_lengths = [25, 50, 100]

os.makedirs(output_dir, exist_ok=True)

imdb_dataset = pd.read_csv(dataset)

imdb_dataset["review"] = imdb_dataset["review"].str.lower()
imdb_dataset["review"] = imdb_dataset["review"].apply(lambda x: re.sub(r"[^a-zA-Z\s]", "", x))
imdb_dataset["review"] = imdb_dataset["review"].apply(lambda x: re.sub(r"\s+", " ", x).strip())
imdb_dataset["label"] = imdb_dataset["sentiment"].map({"positive": 1, "negative": 0})

training_data = imdb_dataset.iloc[:25000]
testing_data = imdb_dataset.iloc[25000:]
training_reviews = training_data["review"].tolist()
training_labels = training_data["label"].values  
testing_reviews = testing_data["review"].tolist()
testing_labels = testing_data["label"].values  

tokenizer = Tokenizer(num_words=vocabulary_size, oov_token="<UNK>")
tokenizer.fit_on_texts(training_reviews)
training_sequences = tokenizer.texts_to_sequences(training_reviews)  
testing_sequences = tokenizer.texts_to_sequences(testing_reviews)

for seq_len in sequence_lengths:
    training_padded = pad_sequences(training_sequences, maxlen=seq_len, padding="post", truncating="post")
    testing_padded = pad_sequences(testing_sequences, maxlen=seq_len, padding="post", truncating="post")
    pd.DataFrame({
    "sequence": [" ".join(map(str, row)) for row in training_padded],
    "label": training_data["label"].values
}).to_csv(os.path.join(output_dir, f"imdb_train_seq{seq_len}.csv"), index=False)

    pd.DataFrame({
        "sequence": [" ".join(map(str, row)) for row in testing_padded],
        "label": testing_data["label"].values
    }).to_csv(os.path.join(output_dir, f"imdb_test_seq{seq_len}.csv"), index=False)
