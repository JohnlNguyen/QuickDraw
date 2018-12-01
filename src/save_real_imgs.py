import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

channels = 1
img_size = 28
img_w = img_h = img_size
img_shape = (img_size, img_size, channels)

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

def get_all_classes():
    df = pd.DataFrame([], columns=['Image', 'Label'])
    for i, label in enumerate(classes):
        data = np.load('./data/%s.npy' % label) / 255
        data = np.reshape(data, [data.shape[0], img_size, img_size, 1])
        df2 = pd.DataFrame([(row, i) for row in data], columns=['Image', 'Label'])
        df = df.append(df2)
    return df.sample(frac=1)


def save_real_imgs(real_imgs):
    doodle_per_img = 16
    for i in range(real_imgs.shape[0] - doodle_per_img):
        plt.figure(figsize=(5,5))
        for k in range(doodle_per_img):
            plt.subplot(4, 4, k+1)
            plt.imshow(real_imgs.iloc[i + k].reshape((img_size, img_size)), cmap='gray')
            plt.axis('off')
        print("Saving {}".format(i))
        plt.tight_layout()
        plt.show()
        plt.savefig('./real_{}.png'.format(i+1))


data = get_all_classes()
real_imgs = data.sample(200)['Image']
doodle_per_img = 16
for i in range(real_imgs.shape[0] - doodle_per_img):
    plt.figure(figsize=(5,5))
    for k in range(doodle_per_img):
        plt.subplot(4, 4, k+1)
        plt.imshow(real_imgs.iloc[i + k].reshape((img_size, img_size)), cmap='gray')
        plt.axis('off')
    print("Saving {}".format(i))
    plt.tight_layout()
    plt.show()
    plt.savefig('./real_{}.png'.format(i+1))
