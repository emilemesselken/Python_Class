from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras import regularizers
from keras.optimizers import Adam

def createModel(X_shape, class_num):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=X_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
        
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.2))

    model.add(Dense(256, kernel_constraint=maxnorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
        
    model.add(Dense(128, kernel_constraint=maxnorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(class_num))
    model.add(Activation('softmax'))

    return model


def createModel2(X_shape, class_num, l2_loss_lambda=None):
    l2 = None if l2_loss_lambda is None else regularizers.l2(l2_loss_lambda)
    
    model = Sequential([
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2,
               input_shape=X_shape),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.20),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2),
        BatchNormalization(),        
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.30),        
        
        Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2),
        BatchNormalization(),
        Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2),
        BatchNormalization(),    
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.40),        
        
        Flatten(),
        
        Dense(1024, activation='relu', kernel_regularizer=l2),
        Dropout(0.30),
        
        Dense(512, activation='relu', kernel_regularizer=l2),
        Dropout(0.20),
        
        Dense(class_num, activation='softmax')
    ])
    
    return model