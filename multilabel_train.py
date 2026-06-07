import pandas as pd
import re
import numpy as np
import pickle
import random
import sys

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# 1- Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 2- Save output to a text file
output_file = open("multilabel_results_FINAL.txt", "w", encoding="utf-8")
sys.stdout = output_file

# 3- Load + Clean
df = pd.read_csv(r"C:\safe_data\train2.csv")

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
df = df[["comment_text"] + LABELS]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["comment_text"] = df["comment_text"].apply(clean_text)

X = df["comment_text"].values
Y = df[LABELS].values.astype(np.float32)

# 4- Split
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.2, random_state=SEED
)

# Save split for reproducible evaluation
with open("multilabel_val_split_FINAL.pkl", "wb") as f:
    pickle.dump({"X_val": X_val, "Y_val": Y_val}, f)

# 5- Tokenize + Pad
MAX_WORDS = 30000
MAX_LEN = 200
EMBED_DIM = 100

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN, padding="post", truncating="post")
X_val_pad   = pad_sequences(tokenizer.texts_to_sequences(X_val),   maxlen=MAX_LEN, padding="post", truncating="post")

# 6- Build Multi-label BiLSTM
model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=EMBED_DIM),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(len(LABELS), activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(curve="ROC", multi_label=True)]
)
model.summary()

early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

history = model.fit(
    X_train_pad, Y_train,
    validation_data=(X_val_pad, Y_val),
    epochs=5,
    batch_size=128,
    callbacks=[early_stop]
)

# 7- Evaluate
Y_prob = model.predict(X_val_pad)
THRESH = 0.5
Y_pred = (Y_prob >= THRESH).astype(int)

print("\n=== Multi-label Classification Report (threshold=0.5) ===")
for i, lab in enumerate(LABELS):
    print(f"\nLabel: {lab}")
    print(classification_report(Y_val[:, i], Y_pred[:, i], digits=4, zero_division=0))

try:
    auc_macro = roc_auc_score(Y_val, Y_prob, average="macro")
    print("\nROC-AUC macro (6 labels):", auc_macro)
except ValueError as e:
    print("\nROC-AUC could not be computed for all labels:", e)

# 8- Plot Loss
plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.title("Multi-label Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("multilabel_loss_curve_FINAL.png", dpi=300, bbox_inches="tight")
plt.show()

# 9- Save artifacts
model.save("toxic_multilabel_bilstm_FINAL.h5")

with open("tokenizer_multilabel_FINAL.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("\nSaved FINAL artifacts: toxic_multilabel_bilstm_FINAL.h5, tokenizer_multilabel_FINAL.pkl, multilabel_val_split_FINAL.pkl")

output_file.close()


