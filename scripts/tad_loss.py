import tensorflow as tf
from tensorflow import keras
from keras import losses
from keras import callbacks
from keras import saving


@saving.register_keras_serializable(package="hce")
class HeatMapCrossEntropy(losses.Loss):
    def __init__(self, alpha=2, beta=4, weight=1, reduce=True, name="heatmap_ce"):
        super().__init__(name=name, reduction=None)
        self.alpha = alpha
        self.beta = beta
        self.weight = weight
        self.reduce = reduce

    def call(self, target, output):
        output = tf.clip_by_value(output, keras.backend.epsilon(), 1. - keras.backend.epsilon())
        
        mask = tf.math.floor(target)
        
        hce = mask * self.weight * tf.math.pow((1 - output), self.alpha) * tf.math.log(output)
        hce += (1 - mask) * tf.math.pow((1 - target), self.beta) * tf.math.pow(output, self.alpha) * tf.math.log(1 - output)
        
        if self.reduce:
            hce = tf.math.reduce_mean(hce)
        
        return -hce

    def get_config(self):
        config = {
            "alpha": self.alpha,
            "beta": self.beta,
            "weight": self.weight,
            "reduce": self.reduce
        }

        return config

    @classmethod
    def from_config(cls, config):
        alpha = config.pop("alpha")
        beta = config.pop("beta")
        weight = config.pop("weight")
        reduce = config.pop("reduce")

        return cls(alpha, beta, weight, reduce)


class LearningSchedule(callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 100 - 1:
            self.model.optimizer.learning_rate /= 10
            print("learning rate modified to: {}".format(self.model.optimizer.learning_rate.numpy()))

        if epoch == 140 - 1:
            self.model.optimizer.learning_rate /= 10
            print("learning rate modified to: {}".format(self.model.optimizer.learning_rate.numpy()))
