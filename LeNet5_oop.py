import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model, load_model
from keras.initializers import glorot_uniform

class LeNet5():

    def __init__(self, X_train, y_train, X_test, y_test, num_classes, hparameters):
        self.X_train = X_train
        self.y_train = y_train
        assert(len(self.X_train) == len(self.y_train))

        self.X_test = X_test
        self.y_test = y_test
        assert(len(self.X_test) == len(self.y_test))

        self.num_classes = num_classes
        self.hparameters = hparameters

    def LeNet5_architecture(self, input_shape):

        """
        implements the architecture of LeNet5.
        Conv2D -> Avg pool -> Conv2D -> Avg pool -> Conv2D -> Flatten -> Fully Connected -> Output
        """

        # Convolution layer (C1) hyperparameters
        s1 = self.hparameters["s1"]
        f1 = self.hparameters["f1"]
        n1 = self.hparameters["n1"]

        # Average pooling layer(S2) hyperparameters
        s2 = self.hparameters["s2"]
        f2 = self.hparameters["f2"]

        # Convolutional layer (C3) hyperparameters
        s3 = self.hparameters["s3"]
        f3 = self.hparameters["f3"]
        n3 = self.hparameters["n3"]

        # Average pooling layers (S4) hyperparameters
        s4 = self.hparameters["s4"]
        f4 = self.hparameters["f4"]

        # Convolutional layer (C5) hyperparameters
        s5 = self.hparameters["s5"]
        f5 = self.hparameters["f5"]
        n5 = self.hparameters["n5"]

        # Number of outputs
        num_classes = self.num_classes

        X_input = Input(input_shape)
        X = X_input

        # Convolution layer 1
        X = Conv2D(n1, (f1,f1), strides = (s1, s1), padding = 'valid', name = 'C1', kernel_initializer = glorot_uniform(seed = 0))(X)
        # Average pooling
        X = AveragePooling2D(pool_size= (f2,f2), strides = (s2,s2), padding = 'valid', name = 'S2')(X)
        # Activation
        X = Activation('tanh')(X)
        # Convolution layer 2
        X = Conv2D(n3, (f3,f3), strides = (s3, s3), padding = 'valid', name = 'C3', kernel_initializer = glorot_uniform(seed = 0))(X)
        #Average pooling
        X = AveragePooling2D(pool_size= (f4,f4), strides = (s4,s4), padding = 'valid', name = 'S4')(X)
        # Activation
        X = Activation('tanh')(X)
        # Convolutional layer 3
        X = Conv2D(n5, (f5,f5), strides = (s5, s5), padding = 'valid', name = 'C5', kernel_initializer = glorot_uniform(seed = 0))(X)
        # Activation
        X = Activation('tanh')(X)
        # Flatten
        X = Flatten()(X)
        # Fully Connected layer
        X = Dense(num_classes, activation = 'softmax', name = 'FC', kernel_initializer = glorot_uniform(seed = 0))(X)

        #create model
        model = Model(inputs = X_input, outputs = X, name = 'LeNet5')

        return model

    def train(self, model, num_epochs = 70, batch_size = 32, learning_rate = 0.001):

        assert(num_epochs>0 and batch_size>0 and learning_rate>0)

        opt = keras.optimizers.Adam(learning_rate = learning_rate)

        #train
        print('Training model....')
        model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        history = model.fit(self.X_train, self.y_train, epochs = num_epochs, batch_size = batch_size)

        return model, history

    def evaluate(self, model):
        preds = model.evaluate(self.X_test, self.y_test)

        return preds[1]
