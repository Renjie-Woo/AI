from __future__ import division, print_function
import tensorflow as tf 
import numpy as np 
import pprint
import os
import sys
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input
from keras.utils import to_categorical
from keras import regularizers, initializers
from keras.callbacks import TensorBoard, ModelCheckpoint,ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization

pp = pprint.PrettyPrinter()
flags = tf.app.flags
flags.DEFINE_string("cpu", "0", "the CPU to use.")
flags.DEFINE_float("learning_rate", 2.5e-3, "Learning rate.")
flags.DEFINE_integer("batch_size", 200, "No. of batch images.")
flags.DEFINE_integer("save_step", 500, "Interval of saving checkpoints")
flags.DEFINE_string("checkpoint", "checkpoint", "Directory to save the checkpoints")
flags.DEFINE_string("log", "summary", "log")
flags.DEFINE_integer("epoch", 100, "Epoch")

FLAGS = flags.FLAGS

def read(filename):
        with open(filename, 'r') as fin:
                string = [line.strip().split('\t') for line in fin.readlines()]
                X = [map(float, line[:-1]) for line in string]
                Y = [int(line[-1]) for line in string]
        return np.array(X), np.array(Y)

		
def Initial(X, Y):
        assert X.shape[0] == Y.shape[0], 'shape not match' #numpy.shape():return a tuple of ints,which are the lengths of dimensions
        number_all = X.shape[0]
        number_train = int(0.8 * number_all)
        number_test = number_all - number_train
        # shuffling
        mask = np.random.permutation(number_all)
        X = X[mask]
        Y = Y[mask]
        # training data
        mask_train = range(number_train)
        X_train = X[mask_train]
        Y_train = Y[mask_train]
        # testing data
        mask_test = range(number_train, number_all)
        X_test = X[mask_test]
        Y_test = Y[mask_test]
        print('All data shape: ', X.shape)
        print('Train data shape: ', X_train.shape)
        print('Train label shape: ', Y_train.shape)
        print('Test data shape: ', X_test.shape)
        print('Test label shape: ', Y_test.shape)
        return X_train, Y_train, X_test, Y_test

	
		
def add_initializer(model, kernel_initializer = initializers.random_normal(stddev=0.01), bias_initializer = initializers.Zeros()):
	for layer in model.layers:
		if hasattr(layer, "kernel_initializer"):
			layer.kernel_initializer = kernel_initializer
		if hasattr(layer, "bias_initializer"):
			layer.bias_initializer = bias_initializer


def add_regularizer(model, kernel_regularizer = regularizers.l2(), bias_regularizer = regularizers.l2()):
	for layer in model.layers:
		if hasattr(layer, "kernel_regularizer"):
			layer.kernel_regularizer = kernel_regularizer
		if hasattr(layer, "bias_regularizer"):
			layer.bias_regularizer = bias_regularizer
			
def Generate_Chip_Model():
        inputs = Input(shape = (441, )) #90% = 441
        hid1 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(256, activation = 'relu')(inputs)))
        hid2 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(384, activation = 'relu')(hid1)))
        hid3 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(384, activation = 'relu')(hid2)))
        hid4 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(256, activation = 'relu')(hid3)))
        hid5 = Dropout(0.5)(BatchNormalization(axis = 1)(Dense(256, activation = 'relu')(hid4)))
        predicts = Dense(93, activation = 'softmax')(hid5)
		
        model = Model(inputs = inputs, outputs = predicts)
        add_regularizer(model)
        return model
		
		
def main(_):
        pp.pprint(flags.FLAGS.__flags)
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
        if not os.path.isdir(FLAGS.checkpoint):
        	os.mkdir(FLAGS.checkpoint)
        if not os.path.isdir(FLAGS.log):
                os.mkdir(FLAGS.log)
        model = Generate_Chip_Model()
        model.summary()
		
        opt = keras.optimizers.rmsprop(lr = 0.001, decay = 1e-6)
        model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
	
        filename = './AI/data/data_ninty_mul.txt'
        x, y = read(filename)
        x_train, y_train, x_test, y_test = Initial(x, y)
		
        y_train_labels = to_categorical(y_train, num_classes = 93)
        y_test_labels = to_categorical(y_test, num_classes = 93)
        model_path = os.path.join(FLAGS.checkpoint, "weights,hdf5")
        callbacks = [
                ModelCheckpoint(filepath=model_path, monitor="val_acc", save_best_only=True, save_weights_only=True),
                TensorBoard(log_dir=FLAGS.log),
                ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2)
	]
        hist = model.fit(x_train, y_train_labels, epochs=FLAGS.epoch, batch_size=100, validation_data=(x_test, y_test_labels), callbacks=callbacks)

        loss, accuracy = model.evaluate(x_test, y_test_labels, batch_size=100, verbose=1)

		
if __name__ == '__main__':
        tf.app.run()