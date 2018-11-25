import os

classes = ['saxophone',
        'raccoon',
        'piano',
        'panda',
        'leg',
        'headphones',
        'ceiling%20fan',
        'bed',
        'basket',
        'aircraft%20carrier']

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    for cls in classes:
        os.system("curl https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/%s.npy > data/%s.npy" % (cls, cls))