import pickle
import numpy as np
import re
import matplotlib.pyplot as plt
import sys

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, average_precision_score

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# 1- Save evaluation output to a text file
output_file = open("binary_evaluation_results_FINAL.txt", "w", encoding="utf-8")
sys.stdout = output_file

# 2- Settings (must match training)
MAX_LEN = 200

# 3- Load FINAL artifacts
model = load_model("toxic_lstm_model_FINAL.h5")

with open("tokenizer_FINAL.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("val_split_FINAL.pkl", "rb") as f:
    split = pickle.load(f)

X_val = split["X_val"]
y_val = split["y_val"]

# 4- Clean text (same as training)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

X_val_clean = [clean_text(t) for t in X_val]

# 5- Tokenize + Pad
X_val_seq = tokenizer.texts_to_sequences(X_val_clean)
X_val_pad = pad_sequences(X_val_seq, maxlen=MAX_LEN, padding="post", truncating="post")

# 6- Predict probabilities
y_prob = model.predict(X_val_pad).ravel()

# 7- Evaluation at threshold 0.5
y_pred = (y_prob >= 0.5).astype(int)

print("\n=== Classification Report (threshold=0.5) ===")
print(classification_report(y_val, y_pred, digits=4))

print("ROC-AUC:", roc_auc_score(y_val, y_prob))

cm = confusion_matrix(y_val, y_pred)
print("\nConfusion Matrix:\n", cm)

# 8- Threshold tuning
thresholds = [0.3, 0.5, 0.7]
for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    p = precision_score(y_val, y_pred_t)
    r = recall_score(y_val, y_pred_t)
    f1 = f1_score(y_val, y_pred_t)
    print(f"Threshold {t}: Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}")

# 9- Precision-Recall curve + AP
prec, rec, thr = precision_recall_curve(y_val, y_prob)
ap = average_precision_score(y_val, y_prob)
print("Average Precision (AP):", ap)

plt.figure()
plt.plot(rec, prec)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.savefig("binary_precision_recall_curve_FINAL.png", dpi=300, bbox_inches="tight")
plt.show()

# 10- Demo predictions
def predict_comment(text, threshold=0.5):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    prob = float(model.predict(pad)[0][0])
    label = "TOXIC" if prob >= threshold else "NON-TOXIC"
    return label, prob

examples = [
    "You are amazing and I love your work",
    "You are stupid and nobody likes you",
    "Nobody wants you here, just disappear",
    "I disagree with you but I respect your opinion"
]

for e in examples:
    label, prob = predict_comment(e, threshold=0.5)
    print(f"\nText: {e}\nPrediction: {label} (score={prob:.4f})")

output_file.close()



