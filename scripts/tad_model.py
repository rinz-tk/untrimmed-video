import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers


def get_model(width, n_joints, label_count):
    position_input = keras.Input(shape=(width, n_joints, 3))
    pos = layers.Conv2D(32, 3, activation="leaky_relu", padding="same")(position_input)
    pos = layers.GroupNormalization(groups=-1)(pos)
    pos = layers.Conv2D(32, 3, activation="leaky_relu", padding="same")(pos)
    pos = layers.GroupNormalization(groups=-1)(pos)
    pos = layers.MaxPool2D(2, 2, padding="same")(pos)

    pos = layers.Conv2D(64, 3, activation="leaky_relu", padding="same")(pos)
    pos = layers.GroupNormalization(groups=-1)(pos)
    pos = layers.Conv2D(64, 3, activation="leaky_relu", padding="same")(pos)
    pos = layers.GroupNormalization(groups=-1)(pos)
    pos = layers.MaxPool2D(2, 2, padding="same")(pos)

    motion_input = keras.Input(shape=(width, n_joints, 3))
    mt = layers.Conv2D(32, 3, activation="leaky_relu", padding="same")(motion_input)
    mt = layers.GroupNormalization(groups=-1)(mt)
    mt = layers.Conv2D(32, 3, activation="leaky_relu", padding="same")(mt)
    mt = layers.GroupNormalization(groups=-1)(mt)
    mt = layers.MaxPool2D(2, 2, padding="same")(mt)

    mt = layers.Conv2D(64, 3, activation="leaky_relu", padding="same")(mt)
    mt = layers.GroupNormalization(groups=-1)(mt)
    mt = layers.Conv2D(64, 3, activation="leaky_relu", padding="same")(mt)
    mt = layers.GroupNormalization(groups=-1)(mt)
    mt = layers.MaxPool2D(2, 2, padding="same")(mt)

    x = layers.Concatenate()([pos, mt])

    x = layers.Conv2D(64, 1, activation="leaky_relu", padding="same")(x)
    x = layers.GroupNormalization(groups=-1)(x)
    x = layers.GroupNormalization(groups=-1)(mt)

    x = layers.Conv2D(128, 3, activation="leaky_relu", padding="same")(x)
    x = layers.GroupNormalization(groups=-1)(x)
    x = layers.Conv2D(64, 1, activation="leaky_relu", padding="same")(x)
    x = layers.GroupNormalization(groups=-1)(x)
    x = layers.Conv2D(128, 3, activation="leaky_relu", padding="same")(x)
    x = layers.GroupNormalization(groups=-1)(x)
    x = layers.MaxPool2D(2, 2, padding="same")(x)

    x = layers.Conv2D(256, 3, activation="leaky_relu", padding="same")(x)
    x = layers.GroupNormalization(groups=-1)(x)
    x = layers.Conv2D(128, 1, activation="leaky_relu", padding="same")(x)
    x = layers.GroupNormalization(groups=-1)(x)
    x = layers.Conv2D(256, 3, activation="leaky_relu", padding="same")(x)
    x = layers.GroupNormalization(groups=-1)(x)
    x = layers.Conv2D(128, 1, activation="leaky_relu", padding="same")(x)
    x = layers.GroupNormalization(groups=-1)(x)
    x = layers.Conv2D(256, 3, activation="leaky_relu", padding="same")(x)
    x = layers.GroupNormalization(groups=-1)(x)
    x = layers.MaxPool2D(2, 2, padding="same")(x)

    x = layers.AveragePooling2D((1, 2))(x)

    x = layers.Conv2DTranspose(128, (4, 1), strides=(2, 1), activation="leaky_relu", padding="same")(x)
    x = layers.GroupNormalization(groups=-1)(x)
    x = layers.Conv2DTranspose(128, (4, 1), strides=(2, 1), activation="leaky_relu", padding="same")(x)
    x = layers.GroupNormalization(groups=-1)(x)
    x = layers.Conv2DTranspose(128, (4, 1), strides=(2, 1), activation="leaky_relu", padding="same")(x)
    x = layers.GroupNormalization(groups=-1)(x)

    heatmap = layers.Conv2D(label_count, (3, 1), activation="leaky_relu", padding="same")(x)
    heatmap = layers.GroupNormalization(groups=-1)(heatmap)
    heatmap = layers.Conv2D(label_count, (1, 1), activation="sigmoid", padding="same")(heatmap)

    offset = layers.Conv2D(1, (3, 1), activation="leaky_relu", padding="same")(x)
    offset = layers.GroupNormalization(groups=-1)(offset)
    offset = layers.Conv2D(1, (1, 1), activation=None, padding="same")(offset)

    size = layers.Conv2D(1, (3, 1), activation="leaky_relu", padding="same")(x)
    size = layers.GroupNormalization(groups=-1)(size)
    size = layers.Conv2D(1, (1, 1), activation=None, padding="same")(size)

    return keras.Model([position_input, motion_input], [heatmap, offset, size])

