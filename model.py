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

tf.keras.utils.set_random_seed(SEED)

class AugmenterFGSM(keras_cv.layers.BaseImageAugmentationLayer):
    def __init__(self, adv_model, epsilon,  **kwargs):
        super().__init__(**kwargs)
        self.adv_model = adv_model
        self.epsilon = epsilon
    
    def augment_image(self, image, transformations=None, **kwargs):
        adv_image = self.adv_model.generate_adv_image(image, self.epsilon)

        if self.adv_model.backbone == "mit_b0":
            adv_image = tf.image.resize(adv_image, (224, 224))

        adv_image = tf.cast(adv_image, tf.uint8)
        adv_image = adv_image[0]

        return adv_image
    
    def augment_label(self, label, transformations=None, **kwargs):
        return label
    
    def augment_images(self, images, transformations=None, **kwargs):
        return self.augment_image(images)
    
    def augment_labels(self, labels, transformations=None, **kwargs):
        return labels
    
    def augment_segmentation_masks(
        self, segmentation_masks, transformations=None, **kwargs
    ):
        return segmentation_masks
    
    def get_config(self):
        config = {
            "adv_model": self.adv_model,
            "epsilon": self.epsilon,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ImageClassifierFGSM(keras_cv.models.ImageClassifier):
    def __init__(self, *args, **kwargs):
        self.epsilon = kwargs.pop("epsilon", 0)
        super().__init__(*args, **kwargs)
    
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape_orig:
            tape_orig.watch(x)
            y_orig_pred = self(x, training=True)
            loss_orig = self.compiled_loss(y, y_orig_pred)
        
        x_gradient = tape_orig.gradient(loss_orig, x)
        
        x_adv = x + self.epsilon*tf.math.sign(x_gradient)
        with tf.GradientTape() as tape_adv:
            y_adv_pred = self(x_adv, training=True)
            loss_adv = self.compiled_loss(y, y_adv_pred)
            loss = ALPHA*loss_orig + (1-ALPHA)*loss_adv

        trainable_vars = self.trainable_variables
        gradients = tape_adv.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_adv_pred)

        return {m.name: m.result() for m in self.metrics}

class ImageClassifierFGSMFramework():
    def __init__(self, backbone, epsilon=0) -> None:
        self.augment_modules = AUGMENT_MODULES
        self.backbone = backbone
        self.epsilon = epsilon

        self.model = ImageClassifierFGSM(
            backbone=BACKBONES[backbone],
            num_classes=N_CLASS,
            activation=ACTIVATION,
            epsilon=epsilon,
        )
        self.model.compile(
            loss=LOSS,
            optimizer=OPTIMIZER,
            metrics=METRICS
        )
        
    def preprocess_data(self, images, labels, augment=False, epsilon=None):
        labels = tf.one_hot(labels, N_CLASS)
        inputs = {"images": images, "labels": labels}

        if augment:
            augmenter = self.augment_modules.copy()
            
            if epsilon:
                augmenter.append(AugmenterFGSM(self, epsilon))

            augmenter = keras_cv.layers.Augmenter(augmenter)
            inputs = augmenter(inputs)

        if self.backbone == "mit_b0":
            inputs["images"] = tf.image.resize(inputs["images"], (224, 224))

        return inputs['images'], inputs['labels']

    def generate_adv_image(self, image, epsilon, return_pertubations=False):
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
            loss = self.model.compiled_loss(label, prediction)

        gradient = tape.gradient(loss, image)
        perturbations = tf.sign(gradient)
        image_adv = image + epsilon*perturbations

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
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=f"{save_dir}/logs")

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

    def evaluate(self, images, labels, epsilon=None):
        predictions = list()
        for image in images:
            if epsilon:
                image = self.generate_adv_image(image, epsilon)

            image_probs = self.model(image).numpy()
            class_num = np.argmax(image_probs[0])
            predictions.append(class_num)

        report = classification_report(
            labels, predictions, output_dict=True
        )

        return report