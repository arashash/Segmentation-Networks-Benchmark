import numpy as np
import matplotlib.pyplot as plt

def plot_single(history, model):
    plt.plot(history['dice_coef'])
    plt.plot(history['val_dice_coef'])
    plt.title(model+' training result')
    plt.ylabel('Dice Coeffienct')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

results = np.load('results.npy').item()
# print(results)

legends = []
for model, history in results.items():
    # plt.plot(history['dice_coef'])
    plt.plot(history['val_dice_coef'])
    legends += [model]
plt.title('validation results')
plt.ylabel('Dice Coeffienct')
plt.xlabel('epoch')
plt.legend(legends, loc='upper left')
plt.savefig('val_dice_coef.png')
plt.close()


legends = []
for model, history in results.items():
    plt.plot(history['dice_coef'])
    legends += [model]
plt.title('training results')
plt.ylabel('Dice Coeffienct')
plt.xlabel('epoch')
plt.legend(legends, loc='upper left')
plt.savefig('train_dice_coef.png')
plt.close()
