import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pickle
import json

import tensorflow_datasets as tfds
import tensorflow as tf

tf.keras.utils.set_random_seed(SEED)

from model import ImageClassifierFGSMFramework
from config import *
from utils import load_image

def train(backbone: str, output_dir: str, epsilon: float=None):
    model = ImageClassifierFGSMFramework(backbone=backbone, epsilon=epsilon)
    
    train_dataset, test_dataset = tfds.load(
        'rock_paper_scissors',
        as_supervised=True,
        split=["train", "test"],
    )
       

    train_dataset = train_dataset.batch(BATCH_SIZE).map(
        lambda x, y: model.preprocess_data(x, y, augment=True, epsilon=epsilon),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    test_dataset = test_dataset.batch(BATCH_SIZE).map(
        lambda x, y: model.preprocess_data(x, y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    model.train(output_dir, train_dataset, test_dataset)

def evaluate(model_path: str, data_dir: str, report_dir: str, epsilon: float=None):
    with open(model_path, "rb") as f:
        model = pickle.dump(f)
    
    images = list()
    labels = list()
    for filename in os.listdir(data_dir):
        label = filename.split("_")[0]
        image = load_image(
            os.path.join(data_dir, filename),
            backbone=model.backbone
        )
        labels.append(label)
        images.append(image)
    
    report = model.evaluate(images, labels, epsilon=epsilon)
    with open(os.path.join(report_dir, "report.json"), "w") as f:
        json.dump(report)