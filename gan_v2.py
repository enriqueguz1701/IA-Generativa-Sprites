import keras
import tensorflow as tf
import numpy as np

from keras import layers
from keras import ops
import matplotlib.pyplot as plt

# from diffusion.metrics import KID

class GAN_ADA(keras.Model):
    def __init__(self, get_generator, get_discriminator, image_size, kid_image_size, batch_size, noise_size,ema):
        super().__init__()

        # self.augmenter = AdaptiveAugmenter()
        self.generator = get_generator()
        self.ema_generator = keras.models.clone_model(self.generator)
        self.discriminator = get_discriminator()

        self.generator.summary()
        self.discriminator.summary()
        self.image_size = image_size
        self.kid_image_size = kid_image_size
        self.batch_size = batch_size
        self.noise_size = noise_size
        self.ema = ema
        
        self.generator_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.discriminator_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.real_accuracy = keras.metrics.BinaryAccuracy(name="real_acc")
        self.generated_accuracy = keras.metrics.BinaryAccuracy(name="gen_acc")

    def compile(self, generator_optimizer, discriminator_optimizer, **kwargs):
        super().compile(**kwargs)

        # separate optimizers for the two networks
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        # self.generator_loss_tracker = keras.metrics.Mean(name="g_loss")
        # self.discriminator_loss_tracker = keras.metrics.Mean(name="d_loss")
        # self.real_accuracy = keras.metrics.BinaryAccuracy(name="real_acc")
        # self.generated_accuracy = keras.metrics.BinaryAccuracy(name="gen_acc")
        
        # self.augmentation_probability_tracker = keras.metrics.Mean(name="aug_p")
        # self.kid = KID(name="kid", image_size=self.image_size, kid_image_size=self.kid_image_size, batch_size=self.batch_size)

    @property
    def metrics(self):
        return [
            self.generator_loss_tracker,
            self.discriminator_loss_tracker,
            self.real_accuracy,
            self.generated_accuracy,
            # self.augmentation_probability_tracker,
            # self.kid,
        ]

    def generate(self, batch_size, training):
        latent_samples = tf.random.normal(shape=(self.  batch_size, self.noise_size))
        # use ema_generator during inference
        if training:
            generated_images = self.generator(latent_samples, training=training)
        else:
            generated_images = self.ema_generator(latent_samples, training=training)
        return generated_images

    def call(self, inputs, training=False):
        generated_images = self.generate(self.batch_size, training)
        return self.discriminator(generated_images, training=training)
    
    def adversarial_loss(self, real_logits, generated_logits):
        # this is usually called the non-saturating GAN loss

        real_labels = tf.ones(shape=(self.batch_size, 1))
        generated_labels = tf.zeros(shape=(self.batch_size, 1))

        # the generator tries to produce images that the discriminator considers as real
        generator_loss = keras.losses.binary_crossentropy(
            real_labels, generated_logits, from_logits=True
        )
        # the discriminator tries to determine if images are real or generated
        discriminator_loss = keras.losses.binary_crossentropy(
            tf.concat([real_labels, generated_labels], axis=0),
            tf.concat([real_logits, generated_logits], axis=0),
            from_logits=True,
        )

        return tf.reduce_mean(generator_loss), tf.reduce_mean(discriminator_loss)

    def train_step(self, real_images):
        # Se calculan los gradientes dos veces, porque se calculan para el generador y el discriminador
        with tf.GradientTape(persistent=True) as tape:
            # Se generan imágenes falsas usando el generador
            generated_images = self.generate(self.batch_size, training=True)
            # Se obtienen los valores de salida del discriminador para imágenes reales y generadas
            real_logits = self.discriminator(real_images, training=True)
            generated_logits = self.discriminator(generated_images, training=True)

            # Se calculan las pérdidas usando el método adversarial_loss
            generator_loss, discriminator_loss = self.adversarial_loss(
                real_logits, generated_logits
            )

        # Se calculan los gradientes de la pérdida con respecto a los pesos entrenables del generador
        # y del discriminador
        generator_gradients = tape.gradient(
            generator_loss, self.generator.trainable_weights
        )
        discriminator_gradients = tape.gradient(
            discriminator_loss, self.discriminator.trainable_weights
        )
        
        # Se actualizan los pesos
        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_weights)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_weights)
        )

        # Se actualizan las métricas
        self.generator_loss_tracker.update_state(generator_loss)
        self.discriminator_loss_tracker.update_state(discriminator_loss)
        self.real_accuracy.update_state(1.0, self.step(real_logits))
        self.generated_accuracy.update_state(0.0, self.step(generated_logits))

        # Se usa una media móvil exponencial para los pesos del generador, mejorando la estabilidad
        # del entrenamiento y reducir la varianza en la calidad de las imágenes generadas
        for weight, ema_weight in zip(
            self.generator.weights, self.ema_generator.weights
        ):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # Se devuelve un diccionario con las métricas de entrenamiento
        return {
            'g_loss' : self.generator_loss_tracker.result(),
            'd_loss' : self.discriminator_loss_tracker.result(),
            'real_acc' : self.real_accuracy.result(),
            'gen_acc' : self.generated_accuracy.result()
        }

    # "hard sigmoid", useful for binary accuracy calculation from logits
    def step(self, values):
        # negative values -> 0.0, positive values -> 1.0
        return 0.5 * (1.0 + tf.sign(values))

    def test_step(self, real_images):
        generated_images = self.generate(self.batch_size, training=False)

        real_logits = self.discriminator(real_images, training=True)
        generated_logits = self.discriminator(generated_images, training=True)
        
        self.real_accuracy.update_state(1.0, self.step(real_logits))
        self.generated_accuracy.update_state(0.0, self.step(generated_logits))
        
        return {
            'real_acc' : self.real_accuracy.result(),
            'gen_acc' : self.generated_accuracy.result()
        }
        
        # self.kid.update_state(real_images, generated_images)

        # # only KID is measured during the evaluation phase for computational efficiency
        # return {self.kid.name: self.kid.result()}

    def plot_images(self, epoch=None, logs=None, num_rows=3, num_cols=6, interval=5):
        # plot random generated images for visual evaluation of generation quality
        if epoch is None or (epoch + 1) % interval == 0:
            num_images = num_rows * num_cols
            generated_images = self.generate(num_images, training=False)

            plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
            for row in range(num_rows):
                for col in range(num_cols):
                    index = row * num_cols + col
                    plt.subplot(num_rows, num_cols, index + 1)
                    plt.imshow(generated_images[index])
                    plt.axis("off")
            plt.tight_layout()
            plt.show()
            plt.close()