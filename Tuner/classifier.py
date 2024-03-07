from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import tensorflow as tf
import numpy as np
import time
import sys
from tensorflow.keras import mixed_precision

class Classifier:
    def __init__(self, num_classes, feature_dim,
                train_feature_path, train_label_path,
                test_feature_path, test_label_path, lb, loop):

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.train_x = np.zeros((0, feature_dim), dtype=np.float16)
        self.train_y = np.zeros((0), dtype=np.int64)
        self.test_x = np.zeros((0, feature_dim), dtype=np.float16)
        self.test_y = np.zeros((0), dtype=np.int64)
        self.init_lr = 0.01
        self.lr_list = np.array([0.015 , 0.012 , 0.0075, 0.0045, 0.0015])
        self.lr_list = self.lr_list * ((lb-(loop/2))/lb) # learning rate scheduling
        self.epoch = 30
        self.batch = 1024
        jet_number = 0
        for path in train_feature_path:
            with open(path, "rb") as frp:
                jet_number += 1
                data = frp.read()
                num_data = len(data) // (2*feature_dim)
                self.train_x = np.vstack((self.train_x, np.frombuffer(data, dtype=np.float16).reshape(num_data, self.feature_dim)))
        for path in test_feature_path:
            with open(path, "rb") as frp:
                data = frp.read()
                num_data = len(data) // (2*feature_dim)
                self.test_x = np.vstack((self.test_x, np.frombuffer(data, dtype=np.float16).reshape(num_data, self.feature_dim)))
        for path in train_label_path:
            with open(path, "rb") as frp:
                self.train_y = np.concatenate((self.train_y, np.frombuffer(frp.read(), dtype=np.uint32)))
        for path in test_label_path:
            with open(path, "rb") as frp:
                self.test_y = np.concatenate((self.test_y, np.frombuffer(frp.read(), dtype=np.uint32)))


    def scheduler(self, epoch, lr):
        if epoch < self.epoch*0.2:
            return self.lr_list[0]
        elif epoch < self.epoch*0.4:
            return self.lr_list[1]
        elif epoch < self.epoch*0.6:
            return self.lr_list[2]
        elif epoch < self.epoch*0.8:
            return self.lr_list[3]
        else:
            return self.lr_list[4]


    def train(self, model_path):
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        regularizer = tf.keras.regularizers.l2(0.002)
        start = time.perf_counter()
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            if model_path == None:
                model = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(self.feature_dim)),
                    tf.keras.layers.Dense(100, activation=None,
                        kernel_regularizer=regularizer,
                        bias_regularizer=regularizer,
                        activity_regularizer=regularizer),
                    tf.keras.layers.Softmax(activity_regularizer=regularizer)
                ])
                optimizer = tf.keras.optimizers.legacy.Adam(lr=self.init_lr, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
                model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            else:
                model = tf.keras.models.load_model(model_path)
                optimizer = tf.keras.optimizers.legacy.Adam(lr=self.init_lr, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
                model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            callback = tf.keras.callbacks.LearningRateScheduler(self.scheduler)
            model.fit(self.train_x, self.train_y, batch_size=self.batch, epochs=self.epoch, callbacks=[callback], verbose=0)

            loss, accuracy = model.evaluate(self.test_x, self.test_y, verbose=0)

        end = time.perf_counter()

        print("Training time:", end - start)
        save_path = "./save_model"
        model.save(save_path)

        return save_path, accuracy
