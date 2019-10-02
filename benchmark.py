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
models = ['FPN', 'Unet']
encoders = ['vgg16', 'resnet18', 'seresnet18',
            'resnext50', 'seresnext50',
            'densenet121', 'inceptionv3',
            'mobilenet']
for model_name in models:
    for backbone in encoders:
        name = backbone + '_' + model_name
        print(name)
        model = get_backboned_model(model_name, backbone)
        history = train(model, X, Y)
        results[name] = history.history
np.save('results.npy', results)
# summarize(results)
