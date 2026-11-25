"""Train a MobileNetV2 transfer-learning image classifier.

Usage:
    python train.py                              # downloads the TF Flowers dataset (5 classes)
    python train.py --data_dir path/to/images    # your own dataset: one sub-folder per class
"""
import argparse
import json
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from src.model import IMG_SIZE, build_model, unfreeze_top

FLOWERS_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=None, help="folder with one sub-folder per class")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10, help="epochs for the frozen-base phase")
    p.add_argument("--fine_tune_epochs", type=int, default=5)
    p.add_argument("--fine_tune_layers", type=int, default=30)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="models")
    p.add_argument("--reports", default="reports")
    return p.parse_args()


def find_class_root(path: Path) -> Path:
    """Walk down single-folder wrappers until we reach the folder that holds the class folders."""
    while True:
        subdirs = [d for d in path.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if len(subdirs) == 1:
            path = subdirs[0]
        else:
            return path


def get_data_dir(arg):
    if arg:
        return find_class_root(Path(arg))
    archive = keras.utils.get_file("flower_photos", origin=FLOWERS_URL, extract=True)
    return find_class_root(Path(archive))


def plot_history(histories, path):
    acc = sum((h.history["accuracy"] for h in histories), [])
    val_acc = sum((h.history["val_accuracy"] for h in histories), [])
    loss = sum((h.history["loss"] for h in histories), [])
    val_loss = sum((h.history["val_loss"] for h in histories), [])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (tr, va, name) in zip(axes, [(acc, val_acc, "Accuracy"), (loss, val_loss, "Loss")]):
        ax.plot(tr, label="train")
        ax.plot(va, label="validation")
        if len(histories) > 1:
            ax.axvline(len(histories[0].history["loss"]) - 0.5, ls="--", c="gray", label="start fine-tuning")
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir, rep_dir = Path(args.out), Path(args.reports)
    out_dir.mkdir(exist_ok=True)
    rep_dir.mkdir(exist_ok=True)
    keras.utils.set_random_seed(args.seed)

    data_dir = get_data_dir(args.data_dir)
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=args.val_split,
        subset="both",
        seed=args.seed,
        image_size=IMG_SIZE,
        batch_size=args.batch_size,
    )
    class_names = train_ds.class_names
    print("Classes:", class_names)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    model, base = build_model(len(class_names))
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    early = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss")

    print("\nPhase 1: training the classification head (base frozen)")
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[early])
    histories = [h1]

    if args.fine_tune_epochs > 0:
        print(f"\nPhase 2: fine-tuning the top {args.fine_tune_layers} layers of MobileNetV2")
        unfreeze_top(base, args.fine_tune_layers)
        model.compile(
            optimizer=keras.optimizers.Adam(1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs, callbacks=[early])
        histories.append(h2)

    plot_history(histories, rep_dir / "training_curves.png")

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    # Collect labels and predictions in the same pass so they stay aligned.
    y_true, y_pred = [], []
    for images, labels in val_ds:
        y_true.append(labels.numpy())
        y_pred.append(np.argmax(model.predict_on_batch(images), axis=1))
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    (rep_dir / "classification_report.txt").write_text(report)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_names, cmap="Blues", xticks_rotation=45
    )
    disp.figure_.tight_layout()
    disp.figure_.savefig(rep_dir / "confusion_matrix.png", dpi=150)

    model.save(out_dir / "mobilenetv2_classifier.keras")
    (out_dir / "class_names.json").write_text(json.dumps(class_names))
    metrics = {
        "val_accuracy": round(float(val_acc), 4),
        "val_loss": round(float(val_loss), 4),
        "head_phase_best_val_accuracy": round(float(max(h1.history["val_accuracy"])), 4),
        "epochs_head": len(h1.history["loss"]),
        "epochs_fine_tune": len(histories[1].history["loss"]) if len(histories) > 1 else 0,
        "classes": class_names,
    }
    (rep_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
