import segmentation_models as sm
from keras.layers import Input, Conv2D
from keras.models import Model


def get_backboned_model(model, backbone, freeze=True):

    if model == 'Unet':
        base_model = sm.Unet(
            backbone_name=backbone, encoder_weights='imagenet', classes=1, activation='sigmoid', freeze_encoder=freeze)
    elif model == 'FPN':
        base_model = sm.FPN(
            backbone_name=backbone, encoder_weights='imagenet', classes=1, activation='sigmoid', freeze_encoder=freeze)
    elif model == 'Linknet':
        base_model = sm.Linknet(
            backbone_name=backbone, encoder_weights='imagenet', classes=1, activation='sigmoid', freeze_encoder=freeze)
    elif model == 'PSPNet':
        base_model = sm.PSPNet(
            backbone_name=backbone, encoder_weights='imagenet', classes=1, activation='sigmoid', freeze_encoder=freeze)
    else:
        print('Model not identified! Unet is selected')
        base_model = sm.Unet(
            backbone_name=backbone, encoder_weights='imagenet', classes=1, activation='sigmoid', freeze_encoder=freeze)

    inp = Input(shape=(96, 96, 1))
    l1 = Conv2D(3, (1, 1))(inp)  # map N channels data to 3 channels
    out = base_model(l1)

    model = Model(inp, out, name=base_model.name)

    # print(model.summary())
    return model
