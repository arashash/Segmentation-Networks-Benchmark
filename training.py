import numpy as np
from keras import backend as K
from keras.optimizers import Adam
from skimage.transform import resize

img_rows = 96
img_cols = 96

smooth = 1.

def reshape(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

def preprocess(imgs_train, imgs_mask_train):
    print('-' * 30)
    print('Preprocessing data...')
    print('-' * 30)
    imgs_train = reshape(imgs_train)
    imgs_mask_train = reshape(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('X.shape is '+str(imgs_train.shape))
    print('Y.shape is '+str(imgs_mask_train.shape))
    return imgs_train, imgs_mask_train


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def train(model, imgs_train, imgs_mask_train):
    print('-' * 30)
    print('Compiling model...')
    print('-' * 30)
    model.compile(optimizer=Adam(lr=5e-5), loss=dice_coef_loss, metrics=[dice_coef])

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    history = model.fit(imgs_train, imgs_mask_train, batch_size=32, nb_epoch=20,
                        verbose=1, shuffle=True, validation_split=0.2)
    return history
