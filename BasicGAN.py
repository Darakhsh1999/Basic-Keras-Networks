import keras
import os
from keras.optimizers import RMSprop
from keras import layers
from keras import Model
from keras.datasets import cifar10
from keras.utils.image_utils import array_to_img
import numpy as np

latent_dim = 32
height = 32
width = 32
channels = 3


##### Generator #####

generator_input = keras.Input(shape= (latent_dim,))
x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16,16,128))(x)

x = layers.Conv2D(256, 5, padding= 'same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2DTranspose(256, 4, strides= 2, padding= 'same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(256, 5, padding= 'same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding= 'same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(channels, 7, activation= 'tanh', padding= 'same')(x)
generator = Model(generator_input, x)
generator.summary()



##### Discriminator #####

discriminator_input = layers.Input(shape= (height, width, channels))

x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides= 2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides= 2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides= 2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)


x = layers.Dropout(0.4)(x)
x = layers.Dense(1, activation= 'sigmoid')(x)

discriminator = Model(discriminator_input, x)
discriminator.summary()


discriminator_optimizer = RMSprop(lr= 0.0008, clipvalue= 1.0, decay= 1e-8)
discriminator.compile(optimizer= discriminator_optimizer, loss= 'binary_crossentropy')




##### Gan #####
discriminator.trainable = False

gan_input = keras.Input(shape= (latent_dim,))
gan_output = discriminator(generator(gan_input)) # gan_input -> generator -> discriminator -> gan_output
gan = Model(gan_input, gan_output)

gan_optimizer = RMSprop(lr= 0.0004, clipvalue= 1.0, decay= 1e-8)
gan.compile(optimizer= gan_optimizer, loss= 'binary_crossentropy')


##### Gan data #####


(x_train, y_train), (_,_) = cifar10.load_data()

x_train = x_train[y_train.flatten() == 6] # Frog images

x_train = x_train.reshape((x_train.shape[0],) + (height, width, channels)).astype('float32') / 255

iterations = 10000
batch_size = 20
save_dir = 'Gan_images'

##### Gan training #####

start = 0

for step in range(iterations):

    random_latent_vectors = np.random.normal(size= (batch_size, latent_dim))

    generated_images = generator.predict(random_latent_vectors, verbose= 0)


    stop = start + batch_size
    real_images = x_train[start: stop]
    combined_images = np.concatenate([generated_images, real_images])

    labels = np.concatenate([np.ones((batch_size,1)), np.zeros((batch_size,1))])
    labels += 0.05 * np.random.random(labels.shape) # adds noise to labels

    d_loss = discriminator.train_on_batch(combined_images, labels)

    random_latent_vectors = np.random.normal(size= (batch_size, latent_dim))

    misleading_targets = np.zeros((batch_size,1))

    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0
    
    if step % 500 == 0 and step > 0:
        
        print('Discriminator loss: ', d_loss)
        print('Adversarial loss: ', a_loss)

        # Generated image
        img = array_to_img(generated_images[0] * 255.0, scale= False)
        img.save(os.path.join(save_dir,'generated_frog'+str(step)+'.png'))

        # Real image
        img = array_to_img(real_images[0] * 255.0, scale= False)
        img.save(os.path.join(save_dir,'real_frog'+str(step)+'.png'))