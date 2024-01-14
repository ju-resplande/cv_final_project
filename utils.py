import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import numpy as np

from config import CLASSES

def get_label(image_probs):
  image_probs = image_probs.numpy()
  class_num = np.argmax(image_probs[0])
  image_class = CLASSES[class_num]
  class_confidence = image_probs[0][class_num]
  
  return image_class, class_confidence

def load_image(image_path, backbone=None):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image)
    
    dim = 224 if backbone == "mit_b0" else 300
    image = tf.image.resize(image, (dim, dim))
    image = image[None, ...]

    return image

def display_image(image, eps, label, confidence):
  mpl.rcParams['figure.figsize'] = (8, 8)
  mpl.rcParams['axes.grid'] = False
  plt.figure()
  image_to_show = tf.Variable(image[0])
  image_to_show = tf.cast(image_to_show, tf.uint8)

  plt.imshow(image_to_show)
  plt.title(f"epsilon = {eps}\n{label}: {confidence*100:0.3f}% confidence")
  plt.show()