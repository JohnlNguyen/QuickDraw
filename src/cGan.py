
# coding: utf-8

# In[28]:


import argparse
import os
import numpy as np
import math
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt

os.makedirs('images', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

channels = 1
img_size = 28
n_classes = 10
latent_dim = 100
batch_size = 32
learning_rate = .0002
b1 = .5
b2 = .999
sample_interval = 400
n_epochs = 2
img_shape = (img_size, img_size, channels)


cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [  nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim+n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

# Loss functions
adversarial_loss = torch.nn.MSELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()


# In[3]:


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(b1, b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row**2, latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([1 for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, 'images/%d.png' % batches_done, nrow=n_row, normalize=True)


# In[18]:


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

class QuickDrawDataset(Dataset):
    """Quick Draw dataset."""

    def __init__(self, label, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = self.get_all_classes(1000)
        self.label = label
        self.transform = transform

    def get_all_classes(self, sample_per_class=100):
        all_classes = []
        for i, label in enumerate(classes):
            img = np.apply_along_axis(self.__reshape_row, 1, np.load('./data/%s.npy' % label))
            all_classes.extend([(row, i) for row in img[np.random.choice(len(img), sample_per_class)]])
        return all_classes

    def __reshape_row(self, row):
        return np.reshape(row, (28, 28))
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        image = self.data_frame[idx][0]
        label = self.data_frame[idx][1]
        return image, label


# In[19]:


data = QuickDrawDataset(label='panda', transform=transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
dataloader = DataLoader(data, batch_size=32, shuffle=False)


# In[45]:


training_gen_loss = []
training_dis_loss = []

for epoch in range(n_epochs):
    mean_dis_loss = []
    mean_gen_loss = []
    for i, (imgs, labels) in enumerate(dataloader):
        batch_size = len(imgs)

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
        gen_labels = Variable(LongTensor(np.array([1]*batch_size)))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)


        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        mean_dis_loss.append(d_loss.item())
        mean_gen_loss.append(g_loss.item())

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, 1, i, len(dataloader),
                                                            d_loss.item(), g_loss.item()))

        batches_done = epoch * len(dataloader) + i
        if epoch % 50 == 0 and batches_done % sample_interval == 0:
            plt.figure(figsize=(5,5))
            for k in range(4):
                plt.subplot(4, 4, k+1)
                plt.imshow(gen_imgs.detach().numpy()[k][0], cmap='gray')
                plt.axis('off')
            plt.tight_layout()
            plt.savefig('./images/epoch_{}_{}.png'.format(epoch+1, batches_done))

    training_gen_loss.append(np.mean(mean_dis_loss))
    training_dis_loss.append(np.mean(mean_gen_loss))


# In[54]:


def save_model():
    torch.save({
            'epoch': n_epochs,
            'generator': generator.model.state_dict(),
            'discriminator': discriminator.model.state_dict(),
            'optimizer_gen': optimizer_G.state_dict(),
            'optimizer_dis': optimizer_D.state_dict(),
            'G_loss': training_gen_loss,
            'D_loss': training_dis_loss
            }, 'saved_models/models')
save_model()

