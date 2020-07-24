"""
DOCSTRING
"""
import argparse
import glob
import keras
import math
import numpy
import PIL

keras.backend.common.set_image_dim_ordering('th') # ensure our dimension notation matches

class GenerateVideo:
    """
    DOCSTRING
    """
    def __init__(self):
        model = self.generator_model()
        print(model.summary())
        
    def __call__(self):
        self.train(400, 10, False)
        self.generate(1)

    def clean(self, image):
        """
        DOCSTRING
        """
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                if image[i][j]+image[i+1][j]+image[i][j+1]+image[i-1][j]+image[i][j-1] > 127*5:
                    image[i][j] = 255
        return image

    def combine_images(self, generated_images):
        """
        DOCSTRING
        """
        num = generated_images.shape[0]
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
        shape = generated_images.shape[2:]
        image = numpy.zeros((height*shape[0], width*shape[1]), dtype=generated_images.dtype)
        for index, img in enumerate(generated_images):
            i = int(index/width)
            j = index % width
            image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img[0,:,:]
        return image

    def discriminator_model(self):
        """
        DOCSTRING
        """
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(64, (5, 5), input_shape=(1, 128, 128), padding="same"))
        model.add(keras.layers.core.Activation('tanh'))
        model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(4, 4)))
        model.add(keras.layers.convolutional.Convolution2D(128, 5, 5))
        model.add(keras.layers.core.Activation('tanh'))
        model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(2, 2)))
        model.add(keras.layers.core.Flatten())
        model.add(keras.layers.Dense(256))
        model.add(keras.layers.core.Activation('tanh'))
        model.add(keras.layers.Dense(1))
        model.add(keras.layers.core.Activation('sigmoid'))
        return model

    def generate(self, BATCH_SIZE):
        """
        DOCSTRING
        """
        generator = generator_model()
        generator.compile(loss='binary_crossentropy', optimizer="SGD")
        generator.load_weights('goodgenerator.h5')
        noise = numpy.zeros((BATCH_SIZE, 100))
        a = numpy.random.uniform(-1, 1, 100)
        b = numpy.random.uniform(-1, 1, 100)
        grad = (b - a) / BATCH_SIZE
        for i in range(BATCH_SIZE):
            noise[i, :] = numpy.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        #image = combine_images(generated_images)
        print(generated_images.shape)
        for image in generated_images:
            image = image[0]
            image = image * 127.5 + 127.5
            PIL.Image.fromarray(image.astype(numpy.uint8)).save("dirty.png")
            PIL.Image.fromarray(image.astype(numpy.uint8)).show()
            clean(image)
            image = PIL.Image.fromarray(image.astype(numpy.uint8))
            image.show()        
            image.save("clean.png")

    def generator_containing_discriminator(self, generator, discriminator):
        """
        DOCSTRING
        """
        model = keras.models.Sequential()
        model.add(generator)
        discriminator.trainable = False
        model.add(discriminator)
        return model

    def generator_model(self):
        """
        DOCSTRING
        """
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(input_dim=100, output_dim=1024))
        model.add(keras.layers.core.Activation('tanh'))
        model.add(keras.layers.Dense(128*8*8))
        model.add(keras.layers.normalization.BatchNormalization())
        model.add(keras.layers.core.Activation('tanh'))
        model.add(keras.layers.Reshape((128, 8, 8), input_shape=(128*8*8, )))
        model.add(keras.layers.convolutional.UpSampling2D(size=(4, 4)))
        model.add(keras.layers.convolutional.Convolution2D(64, 5, 5, border_mode='same'))
        model.add(keras.layers.core.Activation('tanh'))
        model.add(keras.layers.convolutional.UpSampling2D(size=(4, 4)))
        model.add(keras.layers.convolutional.Convolution2D(1, 5, 5, border_mode='same'))
        model.add(keras.layers.core.Activation('tanh'))
        return model

    def get_args(self):
        """
        DOCSTRING
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", type=str)
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--nice", dest="nice", action="store_true")
        parser.set_defaults(nice=False)
        args = parser.parse_args()
        return args

    def load_data(self, pixels=128, verbose=False):
        """
        DOCSTRING
        """
        print("Loading Data:Started")
        X_train = []
        paths = glob.glob('/logos/*.jpg')
        for path in paths:
            if verbose:
                print(path)
            image = PIL.Image.open(path)
            image = PIL.ImageOps.fit(image, (pixels, pixels), PIL.Image.ANTIALIAS)
            image = PIL.ImageOps.grayscale(image)
            #image.show()
            image = numpy.asarray(image)
            X_train.append(image)
        print("Loading Data:Completed")
        return numpy.array(X_train)

    def train(self, epochs, BATCH_SIZE, weights=False):
        """
        Use this if you have already saved state of the network and want to train it further.
        
        :param epochs: Train for this many epochs
        :param BATCH_SIZE: Size of minibatch
        :param weights: If True, load weights from file, otherwise train the model from scratch. 
        """
        X_train = self.load_data()
        X_train = (X_train.astype(numpy.float32) - 127.5) / 127.5
        X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])
        discriminator = self.discriminator_model()
        generator = self.generator_model()
        if weights:
            generator.load_weights('goodgenerator.h5')
            discriminator.load_weights('gooddiscriminator.h5')
        discriminator_on_generator = self.generator_containing_discriminator(
            generator, discriminator)
        d_optim = keras.optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
        g_optim = keras.optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
        generator.compile(loss='binary_crossentropy', optimizer="SGD")
        discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=g_optim)
        discriminator.trainable = True
        discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
        noise = numpy.zeros((BATCH_SIZE, 100))
        for epoch in range(epochs):
            print("Epoch:", epoch)
            print("Number of batches:", int(X_train.shape[0]/BATCH_SIZE))
            for index in range(int(X_train.shape[0]/BATCH_SIZE)):
                for i in range(BATCH_SIZE):
                    noise[i, :] = numpy.random.uniform(-1, 1, 100)
                image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
                generated_images = generator.predict(noise, verbose=0)
                #print(generated_images.shape)
                if index % 20 == 0 and epoch % 10 == 0:
                    image = combine_images(generated_images)
                    image = image * 127.5 + 127.5
                    destpath = '/logo_generated_images/' + str(epoch) + '_' + str(index) + '.png'
                    PIL.Image.fromarray(image.astype(numpy.uint8)).save(destpath)
                X = numpy.concatenate((image_batch, generated_images))
                y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
                d_loss = discriminator.train_on_batch(X, y)
                print("batch %d d_loss : %f" % (index, d_loss))
                for i in range(BATCH_SIZE):
                    noise[i, :] = numpy.random.uniform(-1, 1, 100)
                discriminator.trainable = False
                g_loss = discriminator_on_generator.train_on_batch(
                    noise, [1] * BATCH_SIZE)
                discriminator.trainable = True
                print("batch %d g_loss : %f" % (index, g_loss))
                if epoch % 10 == 9:
                    generator.save_weights('goodgenerator.h5', True)
                    discriminator.save_weights('gooddiscriminator.h5', True)

if __name__ == '__main__':
   generate_video = GenerateVideo()
   generate_video()
