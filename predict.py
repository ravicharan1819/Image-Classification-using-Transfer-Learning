"""Predict the class of one or more images with the trained model.

Usage:
    python predict.py path/to/image.jpg [more.jpg ...]
"""
import json
import sys
from pathlib import Path

import keras
import numpy as np

from src.model import IMG_SIZE

MODEL_DIR = Path("models")


def load():
    model = keras.models.load_model(MODEL_DIR / "mobilenetv2_classifier.keras")
    class_names = json.loads((MODEL_DIR / "class_names.json").read_text())
    return model, class_names


def predict(model, class_names, path, top_k=3):
    img = keras.utils.load_img(path, target_size=IMG_SIZE)
    x = np.expand_dims(keras.utils.img_to_array(img), 0)  # preprocessing is inside the model
    probs = model.predict(x, verbose=0)[0]
    order = np.argsort(probs)[::-1][:top_k]
    return [(class_names[i], float(probs[i])) for i in order]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    model, names = load()
    for p in sys.argv[1:]:
        preds = predict(model, names, p)
        print(f"{p}: " + ", ".join(f"{c} ({s:.1%})" for c, s in preds))
