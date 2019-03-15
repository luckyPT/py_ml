# coding:utf-8
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, concatenate, Input, Dropout
from keras.models import Model

tf.app.flags.DEFINE_integer('capacity', 30000, 'indicates training epoch')
tf.app.flags.DEFINE_string('model_dir', '../../model', 'indicates model dir')
tf.app.flags.DEFINE_string('train_dir', '../../data/csv/data.csv', 'indicates train data dir')
tf.app.flags.DEFINE_string('val_dir', '../../data/csv/data.csv', 'indicates val data dir')
tf.app.flags.DEFINE_integer('val_sample', 3000, 'indicates val sample count')
tf.app.flags.DEFINE_integer('train_sample', 30000, 'indicates train sample count')
FLAGS = tf.app.flags.FLAGS

model_checkpoint = ModelCheckpoint(FLAGS.model_dir + '/model.{epoch:04d}-{val_loss:.4f}.hdf5', verbose=1)
epochs = 10
batch_size = 2048
steps_per_epoch = FLAGS.train_sample // batch_size
validation_steps = FLAGS.val_sample // batch_size
max_len = 60
sess = tf.Session()

print('tf_version:', tf.__version__)  # 1.10.0
print('keras_version:', keras.__version__)  # 2.2.2


def auc_roc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def extract_csv(filename, batch_size, is_train=True):
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(filename + "*"))
    reader = tf.TextLineReader()
    _, value = reader.read(filename_queue)
    alldata = tf.decode_csv(value, record_defaults=[[0.0] for _ in range(35)], field_delim=",")
    if is_train:
        data = tf.train.shuffle_batch(alldata, batch_size=batch_size, capacity=FLAGS.capacity,
                                      min_after_dequeue=FLAGS.capacity // 2, num_threads=8)
    else:
        data = tf.train.batch(alldata, batch_size=batch_size, num_threads=2)
    return data


train_data = extract_csv(FLAGS.train_dir, batch_size=batch_size)
test_data = extract_csv(FLAGS.val_dir, batch_size=batch_size, is_train=False)
init = (tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)

inputs = Input(shape=(34,))
wide = Dense(1, activation='relu')(inputs)
dense1 = Dense(1000, activation='relu')(inputs)
dropout1 = Dropout(0.2)(dense1)
dense2 = Dense(500, activation='relu')(dropout1)
dropout2 = Dropout(0.2)(dense2)
dense3 = Dense(200, activation='relu')(dropout2)
dropout3 = Dropout(0.2)(dense3)
dense4 = Dense(100, activation='relu')(dropout3)
dropout4 = Dropout(0.2)(dense4)
dense5 = Dense(10, activation='relu')(dropout4)
wide_deep = concatenate([dense5, wide], axis=-1)
prediction = Dense(1, activation='sigmoid')(wide_deep)
model = Model(inputs=inputs, outputs=prediction)

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy', auc_roc])

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


def training(is_train=True):
    while True:
        if is_train:
            dataset = sess.run(train_data)
            label = np.array(dataset[0], dtype='int').ravel()
            feature = np.transpose(dataset[1:])
            yield feature, label
        else:
            dataset = sess.run(test_data)
            label = np.array(dataset[0], dtype='int').ravel()
            feature = np.transpose(dataset[1:])
            yield feature, label


model.fit_generator(training(), epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=training(False),
                    validation_steps=validation_steps, verbose=1, workers=1,
                    callbacks=[model_checkpoint])

coord.request_stop()
coord.join(threads)
K.clear_session()

