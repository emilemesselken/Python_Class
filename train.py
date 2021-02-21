import numpy as np
from PIL import Image
from sklearn.model_selection import GridSearchCV
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from createModel import createModel, createModel2


def train_model(use_online_augmentation, X_train, y_train, X_test, y_test, name, optimizer, loss, epochs=20, seed=21, batch_size=32):

    ###################################
    # Data

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # X_train = (X_train - np.mean(X_train)) / np.std(X_train)
    # X_test = (X_test - np.mean(X_test)) / np.std(X_test)

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    X_shape = X_train.shape[1:]
    class_num = y_test.shape[1]

    ###################################
    # Model

    if use_online_augmentation:
        optimizer = Adam(lr=0.0001, decay=1e-6)

    model = createModel2(X_shape, class_num)
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy", "mean_squared_error"])

    ###################################
    # Training

    # fix random seed for reproducibility
    seed = 10
    np.random.seed(seed)

    # # GRID SEARCH CV
    # batch_size = [10, 20, 40, 60, 80, 100]
    # epochs = [10, 50, 100]
    # learning rate...

    # param_grid = dict(batch_size=batch_size, epochs=epochs)
    # grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, scoring="accuracy")
    # grid_result = grid.fit(X_train, y_train, validation_data=(X_test, y_test))
    
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))

    if use_online_augmentation:
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        datagen = ImageDataGenerator( 
            rotation_range=10,        # randomly rotate between 0-rotation_range angle
            width_shift_range=0.2,    # randomly shift horizontally by this much
            height_shift_range=0.2,   # randomly shift vertically by this much
            shear_range=0.2,          # randomly shear by this much
            zoom_range=0.2,           # randomly zoom (80% - 120%)
            horizontal_flip=True,
            fill_mode='nearest'       # fill any pixels lost in xform with nearest
        )

        num_train_batches = len(X_train) // batch_size
        num_train_batches += (0 if len(X_train) % batch_size == 0 else 1)  

        data = datagen.flow(X_train, y_train, batch_size=batch_size)
        hist = model.fit_generator(data, steps_per_epoch = num_train_batches, validation_data=(X_test, y_test), epochs=epochs)
    else:
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)

    ###################################
    # Save model

    model.save('models/' + str(name) + '.h5')
    model.save_weights('models/' + str(name) + '_weights.h5')

    return hist

