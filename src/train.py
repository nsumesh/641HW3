import os, time, random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from models import build_rnn, build_lstm, build_bilstm

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()
print(f"\n Random seed fixed to: {SEED}")

# Hardware report
device_name = tf.config.list_physical_devices('GPU')
cpu_info = os.popen("sysctl -n machdep.cpu.brand_string").read().strip()
ram_bytes = os.popen("sysctl -n hw.memsize").read().strip()
ram_gb = round(int(ram_bytes) / (1024**3), 2)
print(f"Hardware: {'GPU' if device_name else 'CPU only'} | CPU: {cpu_info} | RAM: {ram_gb} GB\n")


epochs = 10
batch_size = 32
results_dir = "/Users/nsumesh/Documents/GitHub/641HW3/results/metrics.csv"
os.makedirs("results", exist_ok=True)

CONFIGS = [
    ("LSTM","relu","Adam",50,False),
    ("LSTM","relu","Adam",25,False),
    ("LSTM","relu","Adam",100,False),
    ("LSTM","relu","SGD",50,False),
    ("LSTM","relu","RMSprop",50,False),
    ("LSTM","tanh","Adam",50,False),
    ("LSTM","sigmoid","Adam",50,False),
    ("LSTM","relu","Adam",50,True),
    ("LSTM","tanh","Adam",25,False),
    ("LSTM","sigmoid","Adam",100,False),
    ("LSTM","relu","RMSprop",25,False),
    ("LSTM","relu","RMSprop",100,False),
    ("LSTM","tanh","RMSprop",50,False),
    ("LSTM","relu","SGD",25,False),
    ("LSTM","relu","SGD",100,False),
    ("LSTM","tanh","SGD",50,False),
    ("LSTM","relu","Adam",50,True),
    ("LSTM","relu","Adam",100,True),
    ("LSTM","tanh","RMSprop",100,True),
    ("LSTM","relu","SGD",50,True),
    ("RNN","relu","Adam",50,False),
    ("RNN","relu","Adam",25,False),
    ("RNN","relu","Adam",100,False),
    ("RNN","tanh","Adam",50,False),
    ("RNN","sigmoid","Adam",50,False),
    ("RNN","relu","SGD",50,False),
    ("RNN","relu","RMSprop",50,False),
    ("RNN","tanh","RMSprop",50,False),
    ("RNN","relu","Adam",50,True),
    ("RNN","tanh","Adam",25,False),
    ("RNN","sigmoid","Adam",100,False),
    ("BiLSTM","relu","Adam",50,False),
    ("BiLSTM","tanh","Adam",50,False),
    ("BiLSTM","sigmoid","Adam",50,False),
    ("BiLSTM","relu","RMSprop",50,False),
    ("BiLSTM","relu","SGD",50,False),
    ("BiLSTM","tanh","RMSprop",50,False),
    ("BiLSTM","tanh","Adam",25,False),
    ("BiLSTM","tanh","Adam",100,False),
    ("BiLSTM","relu","Adam",100,False),
    ("BiLSTM","relu","Adam",25,False),
    ("BiLSTM","relu","Adam",50,True),
    ("BiLSTM","tanh","Adam",50,True),
    ("BiLSTM","sigmoid","Adam",50,True),
    ("BiLSTM","relu","RMSprop",25,False),
    ("BiLSTM","relu","RMSprop",100,False),
    ("BiLSTM","tanh","RMSprop",100,True),
    ("BiLSTM","relu","SGD",25,False),
    ("BiLSTM","relu","SGD",100,False),
    ("BiLSTM","tanh","SGD",50,True),
]


def load_data(seq_len):
    train = pd.read_csv(f"/Users/nsumesh/Documents/GitHub/641HW3/data/preprocessed/imdb_train_seq{seq_len}.csv")
    test  = pd.read_csv(f"/Users/nsumesh/Documents/GitHub/641HW3/data/preprocessed/imdb_test_seq{seq_len}.csv")

    training_data = np.array([np.fromstring(s, sep=' ') for s in train["sequence"]])
    training_labels = train["label"].values
    testing_data  = np.array([np.fromstring(s, sep=' ') for s in test["sequence"]])
    testing_labels  = test["label"].values
    return training_data, training_labels, testing_data, testing_labels


def run_experiment(model_type, activation, optimizer_name, sequence_len, gradient_clip):
    print(f"\n{model_type:<6} | activation={activation:<7} | optimizer={optimizer_name:<8} | seq={sequence_len:<3} | clip={gradient_clip}")
    training_data, training_labels, testing_data, testing_labels = load_data(sequence_len)

    if model_type == "RNN":
        model = build_rnn(seq_length=sequence_len, activation=activation)
    elif model_type == "LSTM":
        model = build_lstm(seq_length=sequence_len, activation=activation)
    else:
        model = build_bilstm(seq_length=sequence_len, activation=activation)

    if optimizer_name == "Adam":
        optimizer = Adam(clipnorm=1.0) if gradient_clip else Adam()
    elif optimizer_name == "SGD":
        optimizer = SGD(clipnorm=1.0) if gradient_clip else SGD()
    else:
        optimizer = RMSprop(clipnorm=1.0) if gradient_clip else RMSprop()

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    start = time.time()
    model_run = model.fit(training_data, training_labels, epochs=epochs, batch_size=batch_size,validation_data=(testing_data, testing_labels), verbose=0)
    train_time = round(time.time() - start, 2)
    loss_track = pd.DataFrame(model_run.history)
    loss_path = f"results/loss_{model_type}_{activation}_{optimizer_name}_{sequence_len}_{gradient_clip}.csv"
    loss_track.to_csv(loss_path, index=False)
    prediction = (model.predict(testing_data) > 0.5).astype(int).flatten()
    accuracy = np.mean(prediction == testing_labels)
    f1score = f1_score(testing_labels, prediction, average="macro")

    print(f"Accuracy={accuracy:.4f}, F1 Score={f1score:.4f}, Time={train_time:.1f}s")
    return {
        "Model": model_type,
        "Activation": activation,
        "Optimizer": optimizer_name,
        "Seq Length": sequence_len,
        "Grad Clipping": gradient_clip,
        "Accuracy": accuracy,
        "F1 Score": f1score,
        "Epoch Time (s)": train_time,
    }


results = []
for i, (model_type, activation, optimizer, seq, clip) in enumerate(CONFIGS, start=1):
    print(f"\n=== Experiment {i}/{len(CONFIGS)} ===")
    res = run_experiment(model_type, activation, optimizer, seq, clip)
    results.append(res)
    pd.DataFrame([res]).to_csv(
        results_dir,
        mode="a",
        header=not os.path.exists(results_dir),
        index=False,
    )
print("\nAll experiments complete.")
print(f"Results saved to {results_dir}")
