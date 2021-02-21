import json
import matplotlib.pyplot as plt
import numpy as np

def generate_plots(history, title, model_num, plot_num, res):
    plt.figure(plot_num, figsize=(8, 6))
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Nr.{} - Accuracy\n{}\nFinal accuracy: {}%'.format(model_num, title, res))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('figures/{}_accuracy.png'.format(model_num))
    
    plt.figure(plot_num+1, figsize=(8, 6))
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Nr.{} - Loss\n{}\nFinal accuracy: {}%'.format(model_num, title, res))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('figures/{}_loss.png'.format(model_num))

    plt.figure(plot_num+2, figsize=(8, 6))
    # summarize history for mse
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('Model Nr.{} - Mean squared error\n{}\nFinal accuracy: {}%'.format(model_num, title, res))
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('figures/{}_mse.png'.format(model_num))

def generate_data(data):
    # create table of all model results (unused)
    for model in data:
        acc = model[0]["accuracy"]

    dat = data["first"]
    accuracy = dat[0]['accuracy'] # float 0 to 1
    final_train_loss = dat[0]['final_train_loss'] # float > 0
    final_val_loss = dat[0]['final_val_loss'] # float > 0
    f1 = dat[0]['f1'] # float 0 to 1
    precision = dat[0]['precision'] # float 0 to 1
    recall = dat[0]['recall'] # float 0 to 1
    epochs = dat[0]['epochs'] # int > 0
    optimizer_func = dat[0]['optimizer_func'] # string
    loss_func = dat[0]['loss_func'] # string

    data = [[accuracy],
            [final_train_loss],
            [final_val_loss], 
            [f1],
            [precision], 
            [recall],
            [epochs],
            [optimizer_func],
            [loss_func]]
    
    rows = ('accuracy', 'final_train_loss', 'final_val_loss', 'f1', 'precision', 'recall', 'epochs', 'optimizer_func', 'loss_func')
    columns = ['model 1']

    # Add a table at the bottom of the axes
    index = np.arange(len(columns))
    cell_text = []
    for row in range(len(rows)):
        #plt.scatter(index, data[row], label = row)
        cell_text.append([x for x in data])
    plt.table(cellText=cell_text, rowLabels=rows, colLabels=columns)
    plt.show()