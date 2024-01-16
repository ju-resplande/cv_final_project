import keras_cv
import tensorflow as tf

SEED = 812
tf.keras.utils.set_random_seed(SEED)

## data
CLASSES = {
    0: "rock",
    1: "paper",
    2: "scissors",
}

CLASS2ID = {label:idx for idx, label in CLASSES.items()}

N_CLASS = len(CLASSES)

## models
BACKBONES = {
    "mit_b0": keras_cv.models.MiTBackbone.from_preset("mit_b0_imagenet"),
    "efficientnetv2_b0": keras_cv.models.EfficientNetV2Backbone.from_preset("efficientnetv2_b0_imagenet"),
}

## parameters
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=1e-5)

AUGMENT_MODULES = [
    keras_cv.layers.RandomFlip(),
    keras_cv.layers.RandAugment(value_range=(0, 255)),
    keras_cv.layers.CutMix(),
]

METRICS = [
    tf.keras.metrics.CategoricalAccuracy(),
    tf.keras.metrics.F1Score(average="macro"),
]
ACTIVATION = tf.keras.activations.softmax
LOSS = tf.keras.losses.CategoricalCrossentropy()
BATCH_SIZE = 8
EPOCHS = 8
ALPHA = 0.5
