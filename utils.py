import numpy as np
import os
import skimage
import skimage.io
import skimage.transform


DATA_PATH = './Data/aligned'
TRAIN_SET = ['./Data/fold_frontal_0_data.txt',
            './Data/fold_frontal_1_data.txt',
            './Data/fold_frontal_2_data.txt',
            './Data/fold_frontal_3_data.txt']
VAL_SET = ['./Data/fold_frontal_4_data.txt']


def get_data(file_paths):
    # TODO: wasteful iterations through file structure; optimize!
    image_paths, labels = [], []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            for line in f:
                l = line.strip().split('\t')
                folder, part_name = l[0], l[1]
                for dd, _, ff in os.walk(os.path.join(DATA_PATH, folder)):
                    for fff in ff:
                        if fff.find(part_name) != -1:
                            image_path = os.path.join(dd, fff)
                            image_paths.append(image_path)
                            if l[4] == 'f': # female
                                label = 1
                            elif l[4] == 'm': # male
                                label = 0
                            labels.append(label)
                            break
    return np.array(image_paths), np.array(labels)


def load_image(image_path):
    image = skimage.io.imread(image_path)
    image = image.astype(float)
    resized_image = skimage.transform.resize(image, (224, 224))
    return resized_image


def get_mean(images):
    mean = np.zeros(load_image(images[0]).shape)
    for idx, im in enumerate(images):
        alpha = 1 / (idx + 1)
        mean = alpha * load_image(im) + (1 - alpha) * mean
    return mean
