import numpy as np
from data import load_data
from models import get_backboned_model
# from plotting import summarize
from training import train, preprocess
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

X, Y = load_data()
X, Y = preprocess(X, Y)

results = {}
models = ['Unet']
encoders = [
            'vgg16', 'vgg19',
            'resnet18', 'resnet34', 'resnet50', 'resnet101',
            'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101',
            'resnext50', 'resnext101',
            'seresnext50', 'seresnext101',
            'senet154', 'densenet121',
            'mobilenet', 'mobilenetv2'
            ]
for model_name in models:
    for backbone in encoders:
        # name = backbone + '_' + model_name
        name = backbone
        print(name)
        model = get_backboned_model(model_name, backbone, freeze=False)
        history = train(model, X, Y)
        results[name] = history.history
        np.save('results.npy', results)
