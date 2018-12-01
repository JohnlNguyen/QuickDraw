import os

import numpy as np

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Dropout, Flatten
from keras.layers import Activation, Reshape, Conv2DTranspose, UpSampling2D
from keras.optimizers import RMSprop
from keras import optimizers

import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

matplotlib.interactive(True)



channels = 1
img_size = 28
img_w = img_h = img_size
img_shape = (img_size, img_size, channels)
n_epochs = 500

classes = ['saxophone',
    'raccoon',
    'piano',
    'panda',
    'leg',
    'headphones',
    'ceiling_fan',
    'bed',
    'basket',
    'aircraft_carrier']


def discriminator_builder(depth=64,p=0.4):

    # Define inputs
    inputs = Input((img_w,img_h,1))

    # Convolutional layers
    conv1 = Conv2D(depth*1, 5, strides=2, padding='same', activation='relu')(inputs)
    conv1 = Dropout(p)(conv1)

    conv2 = Conv2D(depth*2, 5, strides=2, padding='same', activation='relu')(conv1)
    conv2 = Dropout(p)(conv2)

    conv3 = Conv2D(depth*4, 5, strides=2, padding='same', activation='relu')(conv2)
    conv3 = Dropout(p)(conv3)

    conv4 = Conv2D(depth*8, 5, strides=1, padding='same', activation='relu')(conv3)
    conv4 = Flatten()(Dropout(p)(conv4))

    output = Dense(1, activation='sigmoid')(conv4)

    model = Model(inputs=inputs, outputs=output)

    return model

discriminator = discriminator_builder()
discriminator.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8), metrics=['accuracy'])

def generator_builder(z_dim=100,depth=64,p=0.4):

    # Define inputs
    inputs = Input((z_dim,))

    # First dense layer
    dense1 = Dense(7*7*64)(inputs)
    dense1 = BatchNormalization(axis=-1,momentum=0.9)(dense1)
    dense1 = Activation(activation='relu')(dense1)
    dense1 = Reshape((7,7,64))(dense1)
    dense1 = Dropout(p)(dense1)

    # Convolutional layers
    conv1 = UpSampling2D()(dense1)
    conv1 = Conv2DTranspose(int(depth/2), kernel_size=5, padding='same', activation=None,)(conv1)
    conv1 = BatchNormalization(axis=-1,momentum=0.9)(conv1)
    conv1 = Activation(activation='relu')(conv1)

    conv2 = UpSampling2D()(conv1)
    conv2 = Conv2DTranspose(int(depth/4), kernel_size=5, padding='same', activation=None,)(conv2)
    conv2 = BatchNormalization(axis=-1,momentum=0.9)(conv2)
    conv2 = Activation(activation='relu')(conv2)

    #conv3 = UpSampling2D()(conv2)
    conv3 = Conv2DTranspose(int(depth/8), kernel_size=5, padding='same', activation=None,)(conv2)
    conv3 = BatchNormalization(axis=-1,momentum=0.9)(conv3)
    conv3 = Activation(activation='relu')(conv3)

    # Define output layers
    output = Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(conv3)

    # Model definition
    model = Model(inputs=inputs, outputs=output)

    return model

generator = generator_builder()

def adversarial_builder(z_dim=100):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8), metrics=['accuracy'])
    return model

AM = adversarial_builder()

def make_trainable(net, is_trainable):
    net.trainable = is_trainable
    for l in net.layers:
        l.trainable = is_trainable


def train(df, epochs=2000,batch=128):
    d_loss = []
    a_loss = []
    running_d_loss = 0
    running_d_acc = 0
    running_a_loss = 0
    running_a_acc = 0
    for i in range(1, epochs+1):
        batch_idx = np.random.choice(df.shape[0] ,batch,replace=False)

        real_imgs = np.array([np.reshape(row, (28, 28, 1)) for row in df['Image'].iloc[batch_idx]])
        fake_imgs = generator.predict(np.random.uniform(-1.0, 1.0, size=[batch, 100]))
        x = np.concatenate((real_imgs,fake_imgs))
        y = np.ones([2*batch,1])
        y[batch:,:] = 0
        make_trainable(discriminator, True)
        d_loss.append(discriminator.train_on_batch(x,y))
        running_d_loss += d_loss[-1][0]
        running_d_acc += d_loss[-1][1]
        make_trainable(discriminator, False)

        noise = np.random.uniform(-1.0, 1.0, size=[batch, 100])
        y = np.ones([batch,1])
        a_loss.append(AM.train_on_batch(noise,y))
        running_a_loss += a_loss[-1][0]
        running_a_acc += a_loss[-1][1]

        log_mesg = "%d: [D loss: %f, acc: %f]" % (i, running_d_loss/i, running_d_acc/i)
        log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, running_a_loss/i, running_a_acc/i)
        print(log_mesg)
        noise = np.random.uniform(-1.0, 1.0, size=[16, 100])
        gen_imgs = generator.predict(noise)
        plt.figure(figsize=(5,5))
        for k in range(gen_imgs.shape[0]):
            plt.subplot(4, 4, k+1)
            plt.imshow(gen_imgs[k, :, :, 0], cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.savefig('./images/panda_{}.png'.format(i+1))
    return a_loss, d_loss


def get_all_classes():
    df = pd.DataFrame([], columns=['Image', 'Label'])
    for i, label in enumerate(classes):
        data = np.load('./data/%s.npy' % label) / 255
        data = np.reshape(data, [data.shape[0], img_size, img_size, 1])
        df2 = pd.DataFrame([(row, i) for row in data], columns=['Image', 'Label'])
        df = df.append(df2)
    return df.sample(frac=1) # shuffle

def get_class(label):
    df = pd.DataFrame([], columns=['Image', 'Label'])
    data = np.load('./data/%s.npy' % label) / 255
    data = np.reshape(data, [data.shape[0], img_size, img_size, 1])
    df2 = pd.DataFrame([(row, i) for row in data], columns=['Image', 'Label'])
    return df.sample(frac=1) # shuffle

def save_model(model_json, name):
    with open(name, "w+") as json_file:
        json_file.write(model_json)

data = get_class(panda)

train(data, epochs=n_epochs, batch=128)


save_model(generator.to_json(), "generator.json")
save_model(AM.to_json(), "discriminator.json")







