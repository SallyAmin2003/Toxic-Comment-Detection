import pandas as pd
import re
import numpy as np
import pickle
import random
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1- Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 2- Save training output to a log file
output_file = open("binary_train_log_FINAL.txt", "w", encoding="utf-8")
sys.stdout = output_file

# 3- Settings
DATA_PATH = r"C:\safe_data\train2.csv"
MAX_WORDS = 30000
MAX_LEN = 200
EMBED_DIM = 100

# 4- Load + Clean
df = pd.read_csv(DATA_PATH)
df = df[["comment_text", "toxic"]]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["comment_text"] = df["comment_text"].apply(clean_text)

X = df["comment_text"].values
y = df["toxic"].values

# 5- Split (save split so evaluation is consistent)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

with open("val_split_FINAL.pkl", "wb") as f:
    pickle.dump({"X_val": X_val, "y_val": y_val}, f)

print("Train size:", len(X_train))
print("Val size:", len(X_val))
print("Train toxic rate:", float(np.mean(y_train)))
print("Val toxic rate:", float(np.mean(y_val)))

# 6- Tokenize + Pad
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
X_val_pad   = pad_sequences(X_val_seq, maxlen=MAX_LEN, padding="post", truncating="post")

print("X_train_pad shape:", X_train_pad.shape)
print("X_val_pad shape:", X_val_pad.shape)

# 7- Class weights (handle imbalance)
classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
print("Class weights:", class_weights)

# 8- Build model
model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=EMBED_DIM),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

# 9- Train
history = model.fit(
    X_train_pad, y_train,
    validation_data=(X_val_pad, y_val),
    epochs=6,
    batch_size=128,
    class_weight=class_weights,
    callbacks=[early_stop]
)

# 10- Save graphs
plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("binary_loss_curve_FINAL.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure()
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.legend()
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("binary_accuracy_curve_FINAL.png", dpi=300, bbox_inches="tight")
plt.show()

# 11- Save model + tokenizer
model.save("toxic_lstm_model_FINAL.h5")

with open("tokenizer_FINAL.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("\nSaved FINAL artifacts: toxic_lstm_model_FINAL.h5, tokenizer_FINAL.pkl, val_split_FINAL.pkl")

output_file.close()


