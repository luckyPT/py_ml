# coding=utf-8

import sklearn.datasets as datasets
import tensorflow as tf
import numpy as np
import os

# windows 报错如下，需要将toco_from_protos.exe所在目录加入到环境变量中
a = b"'toco_from_protos' \xb2\xbb\xca\xc7\xc4\xda\xb2\xbf\xbb\xf2\xcd\xe2\xb2\xbf\xc3\xfc\xc1\xee\xa3\xac\xd2\xb2\xb2\xbb\xca\xc7\xbf\xc9\xd4\xcb\xd0\xd0\xb5\xc4\xb3\xcc\xd0\xf2\r\n\xbb\xf2\xc5\xfa\xb4\xa6\xc0\xed\xce\xc4\xbc\xfe\xa1\xa3\r\n"
print(a.decode("gbk"))
"""
此代码在ubuntu上可以正常运行，如遇见 b'/bin/sh: 1: toco_from_protos: not found\n'
需要在装有tensorflow的Python bin目录下找到toco_from_protos 这个命令，然后在/bin/文件夹下面建立
一个对应的软连接即可。

但是此代码在win10上会遇见错误，详见：
https://github.com/tensorflow/tensorflow/issues/22617
https://github.com/tensorflow/tensorflow/issues/22897
"""

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
iris = datasets.load_iris()
x = iris.data  # we only take the first two features.
y = iris.target
input = tf.placeholder(name="input", dtype=tf.float32, shape=(None, 4))
weight = tf.get_variable(name="weights", dtype=tf.float32, shape=(4, 3))
bias = tf.get_variable(name="bias", dtype=tf.float32, shape=(3,))
output = tf.add(tf.matmul(input, weight), bias, name="output")
loss = tf.losses.sparse_softmax_cross_entropy(tf.constant(np.array(y).reshape([-1, 1])), output)
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        sess.run(train_step, feed_dict={input: np.array(x)})
        if i % 500 == 0:
            print(sess.run(loss, feed_dict={input: np.array(x)}))

    converter = tf.contrib.lite.TFLiteConverter.from_session(sess, [input], [output])
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)
    print('save model complete ... ...')

# -------Test--------
print('start test... ...')
interpreter = tf.contrib.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
for data, label in zip(x, y):
    input_data = np.array(data, dtype=np.float32).reshape(input_shape)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(np.where(output_data == np.max(output_data, axis=1))[1], label)
