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

class ImageClassifierFGSM(keras_cv.models.ImageClassifier):
    def __init__(self, *args, **kwargs):
        self.epsilon = kwargs.pop("epsilon", None)
        super().__init__(*args, **kwargs)
    
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape_orig:
            tape_orig.watch(x)
            y_orig_pred = self(x, training=True)
            loss_orig = self.compiled_loss(y, y_orig_pred)
        
        x_gradient = tape_orig.gradient(loss_orig, x)
  
        if self.epsilon != None:
            epsilon = self.epsilon
        else:
            epsilon = tf.random.uniform((), 0, 1)
            
        
        x_adv = x + epsilon*tf.math.sign(x_gradient)
        
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
    def __init__(self, backbone, epsilon, uniform) -> None:
        self.backbone = backbone
        self.epsilon = epsilon
        self.uniform = uniform

        self.model = ImageClassifierFGSM(
            backbone=BACKBONES[backbone],
            num_classes=N_CLASS,
            activation=ACTIVATION,
            epsilon=epsilon,
            uniform=uniform,
        )
        self.model.compile(
            loss=LOSS,
            optimizer=OPTIMIZER,
            metrics=METRICS
        )
        
    def preprocess_data(self, images, labels, augment=False):
        labels = tf.one_hot(labels, N_CLASS)
        inputs = {"images": images, "labels": labels}

        if augment:
            augmenter = keras_cv.layers.Augmenter(AUGMENT_MODULES)
            inputs = augmenter(inputs)

        if self.backbone == "mit_b0":
            inputs["images"] = tf.image.resize(inputs["images"], (224, 224))

        return inputs['images'], inputs['labels']

    def generate_adv_image(self, image, labels, epsilon, return_pertubations=False):
        self.model.trainable = False

        if len(image.shape) == 3:
            image = image[None, ...]

        labels = tf.convert_to_tensor(labels)
        if len(image.shape) == 0:
            labels = labels[None, ...]

        labels = tf.one_hot(labels, N_CLASS)
        with tf.GradientTape() as tape:
            tape.watch(image)
            prediction = self.model(image)
            loss = self.model.compiled_loss(labels, prediction)

        gradient = tape.gradient(loss, image)
        perturbations = tf.sign(gradient)
        image_adv = image + epsilon*perturbations

        if return_pertubations:
            return image_adv, perturbations
        
        return image_adv
    
    def train(self, train_dataset, save_dir):
        os.makedirs(save_dir)
        
        logger = tf.keras.callbacks.CSVLogger(
            f"{save_dir}/training_log.csv",
            separator=",",
            append=True
        )
        pbar = tfa.callbacks.TQDMProgressBar()
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=f"{save_dir}/logs")
        early_stopper = tf.keras.callbacks.EarlyStopping(monitor='loss')

        self.model.fit(
            train_dataset,
            epochs=EPOCHS,
            verbose=0,
            callbacks=[logger, pbar, tensorboard, early_stopper]
        )

        with open(f"{save_dir}/model.pkl", "wb") as f:
            pickle.dump(self,f)
    
    def predict(self, image):
        self.model.trainable = False
        image_probs = self.model(image)
        image_class, class_confidence = utils.get_label(image_probs)

        return image_class, class_confidence, image_probs

    def evaluate(self, images, labels, epsilon=None):
        self.model.trainable = False
        
        if epsilon:
            images = tf.data.Dataset.from_tensor_slices(images)
            labels2 = tf.data.Dataset.from_tensor_slices(labels)
            data = tf.data.Dataset.zip((images, labels2))
            adv_images = data.batch(BATCH_SIZE).map(
                lambda images, labels2: self.generate_adv_image(images, labels2, epsilon),
                num_parallel_calls=tf.data.AUTOTUNE
            ).prefetch(tf.data.AUTOTUNE)

            images = tf.zeros([])
            for batch in adv_images:
                if not len(images.shape):
                    images = batch
                else:
                    images = tf.concat([images, batch], 0)

        image_probs = self.model(images).numpy()
        predictions = np.argmax(image_probs, axis=1)

        report = classification_report(
            labels, predictions, output_dict=True
        )

        return report