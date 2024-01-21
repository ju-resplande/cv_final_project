import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pickle
import json

import tensorflow_datasets as tfds
from typer import Typer
import tensorflow as tf

from model import ImageClassifierFGSMFramework
from config import *
from utils import load_image

tf.keras.utils.set_random_seed(SEED)

cli = Typer()

@cli.command()
def train(backbone: str, output_dir: str, epsilon: float=0, uniform:str=None):
    model = ImageClassifierFGSMFramework(backbone=backbone, epsilon=epsilon, uniform=uniform)
    
    train_dataset = tfds.load(
        'rock_paper_scissors',
        as_supervised=True,
        split="train",
    )
       
    train_dataset = train_dataset.batch(BATCH_SIZE).map(
        lambda x, y: model.preprocess_data(x, y, augment=True),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    model.train(train_dataset, output_dir)

@cli.command()
def evaluate(model_path: str, data_dir: str, report_path: str, epsilon: float=0):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    images = tf.zeros([])
    labels = list()
    for filename in os.listdir(data_dir):
        label = filename.split("_")[0] #"paper_blabla.jpg"
        label = CLASS2ID[label]
        image = load_image(
            os.path.join(data_dir, filename),
            backbone=model.backbone
        )
        labels.append(label)

        if not len(images.shape):
            images = image
        else:
            images = tf.concat([images, image], 0)

    report = model.evaluate(images, labels, epsilon=epsilon)
    with open(os.path.join(report_path), "w") as f:
        json.dump(report, f)

if __name__ == "__main__":
    cli()