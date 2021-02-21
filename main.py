from keras.datasets import cifar10, fashion_mnist
from keras import backend as K
from train import train_model
from evaluate import evaluate_model
from dashboard import generate_plots, generate_data
from augmentation import start_aug
from createModel import createModel


# SETTINGS
use_best = True # True: Use only the best loss/optimizer combination (adam, mse), False: Use all possible combinations
use_offline_augmentation = False # True: Use data augmentation techniques on test data (pre), False: Unpreprocessed (only standard) train data
use_online_augmentation = False
use_fashion_mnist = True # True: Use fashion mnist, False: Use Cifar10


# DEFINE DIFFERENT TYPES OF MODELS
optimizers = ['SGD', 'RMSprop', 'adam']
losses = ['mean_squared_error', 'mean_squared_logarithmic_error', 'mean_absolute_error', 'categorical_crossentropy']

# create different models based on loss and optimizer combinations
models = []
for opti in optimizers:
    for loss in losses:
        models.append([opti, loss])

# ONLY USE BEST MODEL FROM PREVIOUS TESTS
if use_best:
    models = [['adam', 'categorical_crossentropy']]

# GET DATA
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

if use_fashion_mnist == True:
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    if use_offline_augmentation:
        X_train, y_train = start_aug(X_train, y_train)

    if K.image_data_format() == "channels_first":
        X_train = X_train.reshape((X_train.shape[0], 1, 28, 28))
        X_test = X_test.reshape((X_test.shape[0], 1, 28, 28))
    else:
        X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
        X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

    if use_offline_augmentation: trainX = X_train.reshape(540000, 28, 28, 1)
    else: trainX = X_train.reshape(60000, 28, 28, 1)

    testX = X_test.reshape(10000, 28, 28, 1)

# TRAIN AND EVALUATE MODELS
accuracies = []
hists = []
for idx, model in enumerate(models):
    hist = train_model(use_online_augmentation, X_train, y_train, X_test, y_test, idx, *model)
    hists.append(hist)
    accuracies.append(round(evaluate_model(X_test, y_test, idx), 2))

# GENERATE AND SAVE PLOTS
for idx, model in enumerate(models):
    generate_plots(hists[idx], "Optimizer: {} | Loss function: {}".format(models[idx][0], models[idx][1]), idx, idx*3, accuracies[idx]*100)

print('\nFinal results for all models:\n')
print(accuracies)

