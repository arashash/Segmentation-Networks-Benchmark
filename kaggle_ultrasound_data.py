import os
import numpy as np

from skimage.io import imsave, imread

data_path = '/home/arash/data/'

image_rows = 420
image_cols = 580


def create_data():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = len(images) // 2

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = imread(os.path.join(train_data_path, image_name), as_gray=True)
        img_mask = imread(os.path.join(train_data_path, image_mask_name), as_gray=True)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(data_path+'imgs_train.npy', imgs)
    np.save(data_path+'imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_data():
    imgs_train = np.load(data_path+'imgs_train.npy')
    imgs_mask_train = np.load(data_path+'imgs_mask_train.npy')
    print('X.shape is '+str(imgs_train.shape))
    print('Y.shape is '+str(imgs_mask_train.shape))
    return imgs_train, imgs_mask_train


if __name__ == '__main__':
    create_data()
    # load_data()
