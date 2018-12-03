import os
import random
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Dropout, Flatten
from keras.layers import Activation, Reshape, Conv2DTranspose, UpSampling2D, Embedding, multiply
from keras.optimizers import RMSprop
from keras import optimizers

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import csv

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.makedirs('images', exist_ok=True)
os.makedirs('models', exist_ok=True)


channels = 1
img_size = 28
img_w = img_h = img_size
img_shape = (img_size, img_size, channels)
batch_size = 16  # 512
n_epochs = 2  # 5
max_num_batches = 3  # 100000
batches_needed_before_saving_images = 1  # 500
batches_needed_before_saving_models = 1  # 500
noise_dim = 100
num_saved_images_per_class = 8
classes = [
    #'piano','bee', 'apple'  # for test
    'saxophone',
    'raccoon',
    'piano',
    'panda',
    'leg',
    'headphones',
    'ceiling_fan',
    'bed',
    'basket',
    'aircraft_carrier',
]
num_classes = len(classes)

def discriminator_builder(depth=64,p=0.4):
    # Define inputs
    image = Input(shape=img_shape)
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Embedding(num_classes, np.prod(img_shape))(label)
    reshaped_label_embedding = Reshape(img_shape)(label_embedding)
    inputs = multiply([image, reshaped_label_embedding])

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

    model = Model(inputs=[image, label], outputs=output)

    return model

discriminator = discriminator_builder()
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8),
    metrics=['accuracy'],
)

def generator_builder(z_dim,depth=64,p=0.4):
    # Define inputs
    noise = Input(shape=(z_dim,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = (Embedding(num_classes, z_dim)(label))
    label_embedding = Reshape((z_dim,))(label_embedding)
    inputs = multiply([noise, label_embedding])

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
    model = Model(inputs=[noise, label], outputs=output)

    return model

generator = generator_builder(noise_dim)

def adversarial_builder(z_dim):
    noise = Input(shape=(z_dim,))
    label = Input(shape=(1,), dtype='int32')
    fake_image = generator([noise, label])
    # For the combined model we will only train the generator
    discriminator.trainable = False
    is_real = discriminator([fake_image, label])
    return Model([noise, label], is_real)

AM = adversarial_builder(noise_dim)
AM.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8),
    metrics=['accuracy']
)

def make_trainable(net, is_trainable):
    net.trainable = is_trainable
    for l in net.layers:
        l.trainable = is_trainable

def plot_figures(figures, titles, nrows = 1, ncols=1):
    fig, axeslist = plt.subplots(ncols=ncols+1, nrows=nrows)
    n_titles_ins = 0
    for ind in range(len(figures)):
        if ind % ncols == 0:
            axeslist.ravel()[ind + n_titles_ins].text(0, 0, titles[ind], fontsize=8)
            axeslist.ravel()[ind + n_titles_ins].set_axis_off()
            n_titles_ins += 1
        axeslist.ravel()[ind + n_titles_ins].imshow(figures[ind], cmap='gray')
        axeslist.ravel()[ind + n_titles_ins].set_axis_off()
    # plt.tight_layout()
    return fig

def generate_and_save_images(image_index):
    labels = np.array([
        label for label in range(num_classes)
        for _ in range(num_saved_images_per_class)
    ])
    noise = np.random.uniform(-1.0, 1.0, size=[labels.shape[0], noise_dim])
    gen_imgs = generator.predict([noise, labels])
    fig = plot_figures(
        figures=gen_imgs[:,:,:,0],
        titles=[classes[label] for label in labels],
        nrows=num_classes,
        ncols=num_saved_images_per_class,
    )
    # plt.show()
    fig.savefig("./images/%d.png" % image_index)
    plt.close(fig)

def train(df, n_epochs, batch_size):
    d_loss = []
    a_loss = []
    running_d_loss = 0
    running_d_acc = 0
    running_a_loss = 0
    running_a_acc = 0
    num_processed_batches = 0
    n_batches = len(df) // batch_size
    if n_batches > max_num_batches:
        n_batches = max_num_batches
    with open('training_log.csv', mode='w') as training_log_file:
        fieldnames = ['Epoch', 'BatchIndex', 'BatchSize', 'DLoss', 'DAcc', 'ALoss', 'AAcc']
        training_log = csv.DictWriter(training_log_file, fieldnames=fieldnames)
        training_log.writeheader()
        for epoch in range(n_epochs):
            for batch_ind in range(n_batches):
                # ==================== Extract batch input
                batch_start = batch_ind * batch_size
                batch_end = (batch_ind + 1) * batch_size
                real_imgs = np.array([
                    np.reshape(row, img_shape)
                    for row in df['Image'].iloc[batch_start:batch_end]
                ])
                img_labels = np.array([
                    label
                    for label in df['Label'].iloc[batch_start:batch_end]
                ])

                # ==================== Training Discriminator
                # The latest generator is kept frozen during the discriminator training

                # Creating fake images using the latest generator
                noise = np.random.uniform(-1.0, 1.0, size=[batch_size, noise_dim])
                fake_imgs = generator.predict([noise, img_labels])

                # Preparing the input for discriminator by joining real and fake data
                disc_input_images = np.concatenate(
                    (real_imgs, fake_imgs)
                )
                disc_true_validity = np.concatenate(
                    (np.ones([batch_size,1]), np.zeros([batch_size,1]))
                )
                disc_img_labels = np.concatenate(
                    (img_labels, img_labels)
                )

                # Actual training of the discriminator
                make_trainable(discriminator, True)
                d_loss.append(discriminator.train_on_batch(
                    [disc_input_images, disc_img_labels],
                    disc_true_validity
                ))
                running_d_loss += d_loss[-1][0]
                running_d_acc += d_loss[-1][1]

                # ==================== Training Discriminator
                # The latest discriminator is kept frozen during the generator training
                make_trainable(discriminator, False)

                # Noise for Generator inside AM
                am_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, noise_dim])
                disc_false_validity = np.ones([batch_size,1])
                a_loss.append(AM.train_on_batch(
                    [am_noise, img_labels],
                    disc_false_validity
                ))
                running_a_loss += a_loss[-1][0]
                running_a_acc += a_loss[-1][1]

                # ==================== Logging
                num_processed_batches += 1
                training_record = {
                    'Epoch': epoch,
                    'BatchIndex': batch_ind,
                    'BatchSize': batch_size,
                    'DLoss': running_d_loss/num_processed_batches,
                    'DAcc': running_d_acc/num_processed_batches,
                    'ALoss': running_a_loss/num_processed_batches,
                    'AAcc': running_a_acc/num_processed_batches,
                }
                training_log.writerow(training_record)
                log_mesg = "[Epoch %d / %d, Batch %d / %d]" % (epoch, n_epochs, batch_ind, n_batches)
                log_mesg = "%s: [D loss: %f, acc: %f]" % (
                    log_mesg, training_record['DLoss'], training_record['DAcc']
                )
                log_mesg = "%s  [A loss: %f, acc: %f]" % (
                    log_mesg, training_record['ALoss'], training_record['AAcc']
                )
                print(log_mesg)


                if (num_processed_batches % batches_needed_before_saving_models == 0):
                    save_model(generator.to_json(), "./models/gen%d.json" % num_processed_batches)
                    save_model(AM.to_json(), "./models/am%d.json" % num_processed_batches)
                if (
                    num_processed_batches % batches_needed_before_saving_images == 0
                    or (epoch == n_epochs - 1 and batch_ind >= n_batches - 5)
                ):
                    generate_and_save_images(num_processed_batches)
    return a_loss, d_loss


def get_all_classes():
    df = pd.DataFrame([], columns=['Image', 'Label'])
    for i, label in enumerate(classes):
        data = np.load('./data/%s.npy' % label) / 255
        data = np.reshape(data, [data.shape[0], img_size, img_size, channels])
        df2 = pd.DataFrame([(row, i) for row in data], columns=['Image', 'Label'])
        df = df.append(df2)
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def save_model(model_json, name):
    with open(name, "w+") as json_file:
        json_file.write(model_json)

# def save_real_imgs(real_imgs):
#     doodle_per_img = 16
#     for i in range(real_imgs.shape[0] - doodle_per_img):
#         plt.figure(figsize=(5,5))
#         for k in range(doodle_per_img):
#             plt.subplot(4, 4, k+1)
#             plt.imshow(real_imgs.iloc[i + k].reshape((img_size, img_size)), cmap='gray')
#             plt.axis('off')
#         print("Saving {}".format(i))
#         plt.tight_layout()
#         plt.show()
#         plt.savefig('./images/real_{}.png'.format(i+1))

# ======================= Main Body
print("Loading %d classes..." % num_classes)
data = get_all_classes()
print("Successfully loaded the total of %d images." % data.shape[0])
print("Starting the training with %d full epochs in batches of size %d..." % (n_epochs, batch_size))
train(data, n_epochs, batch_size)
print("Training Finished. Saving models...")
save_model(generator.to_json(), "generator.json")
save_model(AM.to_json(), "adversarial.json")
print("Done!")
