"""MobileNetV2 transfer-learning model with a custom classification head."""
import keras
from keras import layers

IMG_SIZE = (224, 224)


def augmentation() -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def build_model(num_classes: int, weights: str | None = "imagenet", dropout: float = 0.3):
    """Frozen MobileNetV2 feature extractor + GAP + Dropout + Dense softmax head.

    Returns (model, base_model) so the base can be partly unfrozen for fine-tuning.
    """
    base = keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights=weights
    )
    base.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = augmentation()(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)  # scale pixels to [-1, 1]
    x = base(x, training=False)  # keep BatchNorm in inference mode while fine-tuning
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout / 2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="mobilenetv2_classifier"), base


def unfreeze_top(base, n_layers: int = 30):
    """Unfreeze the last `n_layers` of the base (BatchNorm layers stay frozen)."""
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
