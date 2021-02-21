import numpy as np
from keras.models import load_model
from keras.utils import np_utils
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_model(X_test, y_test, name):

    ###################################

    X_test = X_test.astype('float32') / 255.0
    y_test_clean = y_test
    y_test = np_utils.to_categorical(y_test)

    ###################################
    # Model

    model = load_model('models/' + str(name) + '.h5')
    print(model.summary())

    ###################################
    # Evaluation

    scores = model.evaluate(X_test, y_test, verbose=0)
    # print('Accuracy: %.2f%%' % (scores[1]))

    yhat_probs = model.predict(X_test)
    yhat_classes = model.predict_classes(X_test)

    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test_clean, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test_clean, yhat_classes, average='macro')
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test_clean, yhat_classes, average='macro')
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test_clean, yhat_classes, average='macro')
    print('F1 score: %f' % f1)
    
    # ROC AUC
    # auc = roc_auc_score(y_test_clean, yhat_probs)
    # print('ROC AUC: %f' % auc)
    # confusion matrix
    matrix = confusion_matrix(y_test_clean, yhat_classes)
    print(matrix)

    df_cm = pd.DataFrame(matrix, index = [i for i in "ABCDEFGHIJ"], columns = [i for i in "ABCDEFGHIJ"])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.show()

    ###################################
    # Detection
    # TEST INFERENCE SINGLE IMAGE

    inference_image = X_test[1]
    inference_array = np.expand_dims(inference_image, axis=0)

    # im = Image.fromarray(inference_image)
    # im.show()
    # im.save('results/inference_{}.png'.format(name))

    prediction = model.predict(inference_array)
    print(prediction)

    return scores[1]
