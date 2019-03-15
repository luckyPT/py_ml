# coding:utf-8
import numpy as np
import re
import tensorflow as tf
from src.util.object_util import ObjectUtil

tf.enable_eager_execution()

base_path = "path/"
files = ["positive.txt", "negative.txt"]
word2id = ObjectUtil.load_obj(base_path + "/word2id.pickle")
id2word = ObjectUtil.load_obj(base_path + "/word_list.pickle")

BATCH_SIZE = 512
SEQ_LENGTH = 100


def get_data_generator(file_name, label):
    with open(base_path + file_name) as file:
        x = []
        y = []
        lines = []
        for line in file:
            seg_list = line.split("||")
            sample = [0 for _ in range(100)]
            i = 0
            for seg in seg_list:
                if len(seg) > 0 and i < 100 and re.match("[\\u4e00-\\u9fa5]", seg):
                    sample[i] += word2id.get(seg, len(word2id))
                    i = i + 1
            x.append(sample)
            y.append(label)
            lines.append(line)
            if len(y) >= BATCH_SIZE / 2:
                yield x, y, lines
                x = []
                y = []
                lines = []


p_data_generator = get_data_generator(files[0], 1)
n_data_generator = get_data_generator(files[1], 0)

vocabulary_size = len(id2word) + 1  # 字典大小
embedding_size = 50  # 词向量维度

model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(vocabulary_size, embedding_size),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
        tf.keras.layers.Dense(64, activation=tf.nn.elu),
        tf.keras.layers.Dense(32, activation=tf.nn.elu),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ]
)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

for i in range(1000):
    try:
        p_data = p_data_generator.__next__()
        n_data = n_data_generator.__next__()
        features = p_data[0] + n_data[0]
        labels = p_data[1] + n_data[1]
        lines = p_data[2] + n_data[2]
    except StopIteration as e:
        p_data_generator = get_data_generator(files[0], 1)
        n_data_generator = get_data_generator(files[1], 0)
        print("iterator complete...")
        continue
    with tf.GradientTape() as tape:
        pre = model(tf.constant(features), training=True)
        loss = tf.losses.mean_squared_error(tf.reshape(tf.constant(labels), [BATCH_SIZE, 1]), pre)
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())
        error = (pre.numpy().reshape([-1, ]) > 0.5) == np.array(labels)
        error_index = np.where(error == False)[0]
        if error.sum() > 510:
            for e_i in error_index:
                print(e_i, lines[e_i])
        print(i, loss.numpy(), error.sum())  # 迭代次数，损失值，精确率
