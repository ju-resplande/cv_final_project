import keras_cv
import keras

## data
CLASSES = {
    0: "rock",
    1: "paper",
    2: "scissors",
}

N_CLASS = len(CLASSES)

## models
BACKBONES = {
    "csp_darknet": keras_cv.models.CSPDarkNetBackbone.from_preset("csp_darknet_tiny_imagenet"),
    "mit_b0": keras_cv.models.MiTBackbone.from_preset("mit_b0_imagenet"),
    "efficientnetv2_b0": keras_cv.models.EfficientNetV2Backbone.from_preset("efficientnetv2_b0_imagenet"),
}

## parameters
OPTIMIZER = keras.optimizers.Adam(learning_rate=1e-5)

AUGMENT_MODULES = [
    keras_cv.layers.RandomFlip(),
    keras_cv.layers.RandAugment(value_range=(0, 255)),
    keras_cv.layers.CutMix(),
]

EPOCHS = 8
SEED = 812