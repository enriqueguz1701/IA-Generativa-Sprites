import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras import ops
from keras import layers
from tensorflow.keras import backend as K

from tensorflow.keras.utils import deserialize_keras_object

class Sampling(layers.Layer):
    """Uses (mu, log_var) to sample z, the vector encoding a digit."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)
    
    def call(self, inputs):
        # Se obtiene la media y el logaritmo de la varianza de la entrada
        mu, log_var = inputs
        # Se obtiene el batch_size y el tamaño del espacio latente
        batch = ops.shape(mu)[0]
        dim = ops.shape(mu)[1]
        # Se genera ruido aleatorio con las dimensiones del batch_size y del espacio latente 
        # Este ruido sigue una distribución normal estándar (N(0,1)). 
        # Este ruido se usa para añadir variabilidad al espacio latente
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        # ops.exp(0.5 * log_var) -> se calcula la desviación estándar a partir del logaritmo de la varianza
        # La línea entera es la reparametrización z = media + desviación * E
        return mu + ops.exp(0.5 * log_var) * epsilon

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            # Se obtiene la media, el logaritmo de la varianza y un punto del espacio latente
            mu, log_var, z = self.encoder(data)
            # El decodificador toma el punto del espacio latente y genera la reconstrucción 
            # de los datos de entrada
            reconstruction = self.decoder(z)
            # Se calcula la pérdida de reconstrucción (media a lo largo del batch size de imágenes)
            reconstruction_loss = ops.mean(
                ops.sum(
                    # Para calcula la pérdida se utiliza la binary_crossentropy para comparar
                    # el dato original con el dato reconstruido
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis = (1,2)
                )
            )
            # Se calcula la pérdida de regularización
            kl_loss = -0.5 * (1 + log_var - ops.square(mu) - ops.exp(log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis = 1))
            # Se calcula la pérdida total sumando las dos funciones de pérdidas anteriores
            total_loss = reconstruction_loss + kl_loss
        # Se actualizan los pesos
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Se actualizan las métricas
        self.total_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        # La función devuelve un diccionario con las métricas actuales
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }
        
    # Métodos provisionales para guardar el modelo
    def call(self, inputs):
        z = self.encoder(inputs)
        return self.decoder(z)
    
    def get_config(self):
        # Devuelve la configuración del modelo
        config = super(VAE, self).get_config()
        config.update({
            'encoder': self.encoder.get_config(),
            'decoder': self.decoder.get_config(),
        })
        return config
    
    @classmethod
    def from_config(cls, config):
         # Reconstruye el encoder y decoder usando `deserialize_keras_object`
        encoder = deserialize_keras_object(config['encoder'])
        decoder = deserialize_keras_object(config['decoder'])
        return cls(encoder=encoder, decoder=decoder, **config)