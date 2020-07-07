import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from LeNet5_oop import LeNet5
import tensorflow_datasets as tfds


def load_dataset():
    """
    Loads MNIST dataset as numpy array
    """
    X_train, y_train = tfds.as_numpy(tfds.load('mnist', split = 'train', shuffle_files = True , batch_size = -1, as_supervised = True))
    X_test, y_test = tfds.as_numpy(tfds.load('mnist', split = 'test', shuffle_files = True , batch_size = -1, as_supervised = True))

    #zero padding for LeNet 5's input size
    X_train = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    X_test = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

    classes = np.unique(y_train)

    return X_train, y_train, X_test, y_test, classes

def convert_to_one_hot(Y, C):

    """
    Converts labels to one-hot vector
    """

    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def main():

    X_train, y_train, X_test, y_test, classes = load_dataset()
    num_classes = len(classes)
    y_train = convert_to_one_hot(y_train, num_classes).T
    y_test = convert_to_one_hot(y_test, num_classes).T

    (m, n_H0, n_W0, n_C0) = X_train.shape

    hparameters = {"s1": 1, "f1": 5, "n1": 6,
                    "s2":2, "f2":2,
                    "s3":1, "f3": 5, "n3": 16,
                    "s4":2, "f4": 2,
                    "s5":1, "f5": 5, "n5": 120}

    lenet5_network = LeNet5(X_train, y_train, X_test, y_test, num_classes, hparameters)
    model = lenet5_network.LeNet5_architecture(input_shape = (n_H0, n_W0, n_C0) )
    model, history = lenet5_network.train(model, num_epochs = 50, batch_size = 32, learning_rate = 0.001)
    test_accuracy = lenet5_network.evaluate(model)

    print('Accuracy on test set: {:.3f} %'.format(test_accuracy*100))

    plt.plot(history.history['loss'])
    plt.title('Training error')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.show()



main()
