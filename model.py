import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pickle

from sklearn.metrics import classification_report
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
import keras_cv

from config import *
import utils

keras.utils.set_random_seed(SEED)

class AdversarialAugmenter(keras_cv.layers.BaseImageAugmentationLayer):
    def __init__(self, adv_model, sigma,  **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.adv_model = adv_model
    
    def augment_image(self, image, *args, transformation=None, **kwargs):
        adv_image = self.adv_model.generate_adv_image(image, self.sigma)

        if self.backbone == "mit_b0":
            adv_image = tf.image.resize(adv_image, (224, 224))

        adv_image = tf.cast(adv_image, tf.uint8)

        return adv_image

class ImageModel():
    def __init__(self, backbone) -> None:
        self.augment_modules = AUGMENT_MODULES

        self.backbone = backbone

        self.model = keras_cv.models.ImageClassifier(
            backbone=BACKBONES[backbone],
            num_classes=N_CLASS,
            activation="softmax",
        )

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=OPTIMIZER,
            metrics=[
                'accuracy',
                tf.keras.metrics.F1Score(average="macro"),
            ]
        )

    def preprocess_data(self, images, labels, augment=False, sigma=None):
        labels = tf.one_hot(labels, N_CLASS)
        inputs = {"images": images, "labels": labels}

        if augment:
            augmenter = self.augment_modules.copy()
            
            if sigma:
                augmenter.append(AdversarialAugmenter(self, sigma))

            augmenter = keras_cv.layers.Augmenter(augmenter)
            inputs = augmenter(inputs)

        if self.backbone == "mit_b0":
            inputs["images"] = tf.image.resize(inputs["images"], (224, 224))

        return inputs['images'], inputs['labels']

    def generate_adv_image(self, image, sigma, return_pertubations=False):
        self.model.trainable = False

        if len(image.shape) == 3:
            image = image[None, ...]

        image_probs = self.model(image)
        class_num = tf.math.argmax(image_probs[0])

        label = tf.one_hot(class_num, image_probs.shape[-1])
        label = tf.reshape(label, (1, image_probs.shape[-1]))

        with tf.GradientTape() as tape:
            tape.watch(image)
            prediction = self.model(image)
            loss = tf.keras.losses.CategoricalCrossentropy()(label, prediction)

        gradient = tape.gradient(loss, image)
        perturbations = tf.sign(gradient)
        image_adv = image + sigma*perturbations

        if return_pertubations:
            return image_adv, perturbations
        
        return image_adv
    
    def train(self, save_dir, train_dataset, test_dataset):
        os.makedirs(save_dir)
        
        logger = tf.keras.callbacks.CSVLogger(
            f"{save_dir}/training_log.csv",
            separator=",",
            append=True
        )
        pbar = tfa.callbacks.TQDMProgressBar()
        tensorboard = keras.callbacks.TensorBoard(log_dir=f"{save_dir}/logs")

        self.model.fit(
            train_dataset,
            validation_data=test_dataset,
            epochs=EPOCHS,
            verbose=0,
            callbacks=[logger, pbar, tensorboard]
        )

        with open(f"{save_dir}/model.pkl", "wb") as f:
            pickle.dump(self,f)
    
    def predict(self, image):
        self.model.trainable = False
        image_probs = self.model(image)
        image_class, class_confidence = utils.get_label(image_probs)

        return image_class, class_confidence, image_probs

    def evaluate(self, images, labels):
        predictions = list()
        for image in images:
            image_probs = self.model(image).numpy()
            class_num = np.argmax(image_probs[0])
            predictions.append(class_num)

        report = classification_report(
            labels, predictions, output_dict=True
        )

        return report