"""Train the CNN+BiLSTM pose classifier.   Usage: python train.py <dataset_folder>
Dataset layout: dataset/<pose_name>/*.jpg (one folder per pose)."""
import os, sys, json, zipfile, urllib.request
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np, mediapipe as mp
from PIL import Image, ImageOps
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from features import angles, canon_angles, augment, window, T

DATA = sys.argv[1] if len(sys.argv) > 1 else "dataset"
AUG, EPOCHS = 12, 60
TASK = "pose_landmarker_full.task"
URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/" + TASK
rng = np.random.default_rng(42)

def extract():
    """Scan photos safely: progress is saved after every photo, and a photo that crashes the
    scanner (exit code 132/139) is skipped automatically on the next run."""
    if os.path.exists("landmarks.npz"):
        d = np.load("landmarks.npz"); return d["P"], d["W"], d["y"], list(d["labels"])
    if not os.path.exists(TASK): urllib.request.urlretrieve(URL, TASK)
    if os.path.exists("current.txt"):                       # previous run died on this photo
        open("bad.txt", "a").write(open("current.txt").read().strip() + "\n"); os.remove("current.txt")
    bad = set(open("bad.txt").read().split("\n")) if os.path.exists("bad.txt") else set()
    done = set()
    if os.path.exists("scan.jsonl"):
        done = {json.loads(l)["f"] for l in open("scan.jsonl")}
    det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=TASK), running_mode=vision.RunningMode.IMAGE))
    labels = sorted(d for d in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, d)))
    log = open("scan.jsonl", "a")
    for ci, lab in enumerate(labels):
        for root, _, fs in os.walk(os.path.join(DATA, lab)):
            for f in sorted(fs):
                path = os.path.join(root, f)
                if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")) or path in bad or path in done: continue
                open("current.txt", "w").write(path)
                try:
                    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
                    im.thumbnail((1024, 1024))
                    r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.array(im))))
                except Exception:
                    continue
                p = [[l.x * im.width, l.y * im.height] for l in r.pose_landmarks[0]] if r.pose_landmarks else None
                log.write(json.dumps({"f": path, "y": ci, "w": im.width, "p": p}) + "\n"); log.flush()
    if os.path.exists("current.txt"): os.remove("current.txt")
    rows = [json.loads(l) for l in open("scan.jsonl")]
    rows = [r for r in rows if r["p"] is not None]
    P, W, y = np.array([r["p"] for r in rows]), np.array([r["w"] for r in rows]), np.array([r["y"] for r in rows])
    for ci, lab in enumerate(labels): print(f"{lab}: {(y == ci).sum()} usable images")
    print("skipped bad photos:", len(bad))
    np.savez("landmarks.npz", P=P, W=W, y=y, labels=labels)
    return P, W, y, labels

P, W, y, labels = extract()
tr, va = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=1)

def build(ids, n):
    X, Y = [], []
    for i in ids:
        for k in range(n):
            X.append(window(P[i] if k == 0 else augment(P[i], W[i], rng), rng)); Y.append(y[i])
    return np.array(X), np.array(Y)
Xtr, ytr = build(tr, AUG); Xva, yva = build(va, 1)
print("train", Xtr.shape, "val", Xva.shape)

inp = layers.Input((T, Xtr.shape[2]))
x = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)       # CNN: local temporal patterns
x = layers.BatchNormalization()(x)
x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
x = layers.Dropout(0.3)(x)
x = layers.Bidirectional(layers.LSTM(64))(x)                           # LSTM: sequence summary
x = layers.Dropout(0.3)(layers.Dense(64, activation="relu")(x))
model = models.Model(inp, layers.Dense(len(labels), activation="softmax")(x))
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS, batch_size=64, callbacks=[
    callbacks.EarlyStopping(patience=8, restore_best_weights=True), callbacks.ReduceLROnPlateau(patience=3)])
print(classification_report(yva, model.predict(Xva).argmax(1), target_names=labels))

ideal = {}                                # data-driven "ideal" joint angles per pose
for ci, lab in enumerate(labels):
    A = np.array([canon_angles(angles(P[i])) for i in tr if y[i] == ci])
    ideal[lab] = {"median": np.median(A, 0).round(1).tolist(), "std": A.std(0).round(1).tolist()}
json.dump(labels, open("labels.json", "w")); json.dump(ideal, open("ideal.json", "w"))

import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, "model")
with zipfile.ZipFile("pose_model_bundle.zip", "w") as z:
    for f in ["labels.json", "ideal.json"] + ["model/" + f for f in os.listdir("model")]: z.write(f)
print("Done. Copy the contents of pose_model_bundle.zip into your website folder.")