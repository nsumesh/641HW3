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

reviews = imdb_dataset["review"].tolist()
labels = imdb_dataset["label"].values  

tokenizer = Tokenizer(num_words=vocabulary_size, oov_token="<UNK>")
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)  

for seq_len in sequence_lengths:
    padded = pad_sequences(sequences, maxlen=seq_len, padding="post", truncating="post")
    seq_strs = [" ".join(map(str, row)) for row in padded]
    out_df = pd.DataFrame({
        "sequence": seq_strs,
        "label": labels
    })
    out_file = os.path.join(output_dir, f"imdb_seq{seq_len}.csv")
    out_df.to_csv(out_file, index=False)

