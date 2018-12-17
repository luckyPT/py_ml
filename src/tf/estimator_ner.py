# coding:utf-8
import numpy as np
import tensorflow as tf
import pickle

# 参考：
# https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/crf/python/kernel_tests/crf_test.py
# https://github.com/tensorflow/tensorflow/blob/9590c4c32dd4346ea5c35673336f5912c6072bf2/tensorflow/contrib/crf/README.md
# https://www.tensorflow.org/guide/custom_estimators?hl=zh-cn
tf.logging.set_verbosity(tf.logging.INFO)

print('tf_version:', tf.__version__)


class ObjectUtil:
    @staticmethod
    def save_obj(obj, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def load_obj(file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)


batch_size = 64
seq_length = 300
vocabulary_size = 12149
embedding_size = 50


def tensor_scalar(t):
    return tf.reshape(t, [])


def parse_fn(line):
    tensor = tf.sparse_tensor_to_dense(
        tf.string_split(tf.reshape(line, [1]), delimiter='||')
        , default_value='0')

    tensor = tf.map_fn(
        lambda x: tf.sparse_tensor_to_dense(tf.string_split(tf.reshape(x, [1]), delimiter='/'), default_value='0'),
        tf.reshape(tensor, [-1, ]), dtype=tf.string)
    tensor = tf.string_to_number(tensor, out_type=tf.int32)
    feature = tf.reshape(tensor[:, :, 0], [-1, ])
    label = tf.reshape(tensor[:, :, 1], [-1, ])
    feature_padding = tf.pad(feature, paddings=[[seq_length - tf.shape(feature)[0], 0]])
    label_padding = tf.pad(label, paddings=[[seq_length - tf.shape(feature)[0], 0]], constant_values=6)
    return feature_padding, label_padding


prefetch_size = 10240
shuffle_size = 10240


def input_fn(file, batch_size=batch_size, repeat_count=100):
    files = tf.data.Dataset.list_files(file)
    ds = files.interleave(tf.data.TextLineDataset, cycle_length=1) \
        .map(map_func=parse_fn, num_parallel_calls=4).repeat(repeat_count) \
        .shuffle(shuffle_size).batch(batch_size, drop_remainder=True).prefetch(prefetch_size)
    return ds.make_one_shot_iterator().get_next()


def model_fn(features, labels, mode, params):
    # 建立网络模型
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    input_layer = tf.cast(tf.reshape(features, [-1, seq_length]),
                          tf.int32)  # tf.placeholder(tf.int32, [None, seq_length])
    embedding = tf.nn.embedding_lookup(embeddings, input_layer)
    lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=128)
    lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=128)
    ((output_fw, output_bw), (output_state_fw, output_state_bw)) = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw,
                                                                                                   lstm_cell_bw,
                                                                                                   embedding,
                                                                                                   dtype=tf.float32,
                                                                                                   time_major=False)
    lstm_output = tf.concat((output_fw, output_bw), axis=-1)
    dense = tf.layers.dense(units=7, inputs=lstm_output)
    sequence_lengths_t = np.full(features.get_shape()[0], seq_length, dtype=np.int32)
    transition_params = tf.Variable(np.ones(shape=[7, 7]))
    if mode != tf.estimator.ModeKeys.PREDICT:
        labels = tf.reshape(labels, [-1, seq_length])  # tf.placeholder(tf.int32, [None, seq_length])
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(dense, labels, sequence_lengths_t)
        loss = tf.reduce_mean(-log_likelihood)
        train_op = tf.train.AdamOptimizer().minimize(loss, global_step=tf.train.get_global_step())
        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(tf.cast(dense, tf.float64), transition_params,
                                                                    sequence_lengths_t)
        return tf.estimator.EstimatorSpec(mode, predictions={"seq_labels": viterbi_sequence})


# train_data = input_fn("../../data/ner/data.txt").make_one_shot_iterator()
test_data = input_fn("../../data/ner/data.txt")
classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir="/tmp/estimator/model", params={})
# 将test_data传入train方法中会报错，必须直接传：input_fn("/home/panteng/PycharmProjects/sms-ml-py/data/ner/validation_data.txt")
classifier.train(input_fn=lambda: input_fn("../../data/ner/data.txt",
                                           repeat_count=10000), steps=100)

eva = classifier.evaluate(input_fn=lambda: input_fn("../../data/ner/data.txt", batch_size=128,
                                                    repeat_count=10))  # 数据量太少，repeat_count保证至少能有一个batch
print(eva)


# 预测
def in_data():
    data = [[9294, 8818, 8431, 8649, 2975, 1, 7228, 2074, 3526, 2975, 10298, 5639, 11367, 951,
             2975, 6750, 5746, 9477, 7228, 933, 7879, 10368, 3697, 1149, 558, 726, 10381, 9885,
             12020, 897, 5472, 1264, 12020, 1875, 9430, 1875, 2975, 9703, 3697, 1530, 6133, 10069,
             2131, 11158],
            [1530, 4799, 7340, 11158, 7375, 10542, 7914, 6923, 7375, 8818, 7228, 10552, 7228, 9161,
             2074, 3526, 9110, 3141, 5249, 917, 2131, 3911, 7826, 917, 9275, 5289, 951, 1149, 8439,
             7582, 9703, 2975, 10298, 2975, 6750, 8649, 7879, 616, 650, 3883, 3697, 4503, 7879,
             2962, 8649, 2401, 11837, 6750, 9772, 347, 2975, 9708, 11719, 12126, 2131, 8794, 9477,
             8649, 9090, 4374, 2634, 9090, 2457, 635, 2767, 6313, 8649, 10179, 7622, 4999, 4428,
             3526, 9339, 1264, 4541, 12105, 13, 3697, 2131, 5323, 7914, 9110, 3953, 10579, 9963,
             10784, 6765, 11723, 2408, 6126, 6765, 6567, 11094, 11267, 12020, 12051, 8649, 7895,
             1875, 8614, 3697, 2792, 11291, 10822, 4374, 11458, 9257, 572, 1135, 6325, 7914, 11880,
             1264, 5760, 3474, 6635, 3474, 6683, 3697, 2503, 5472, 2001, 2582],
            [1530, 6133, 10069, 2131, 9054, 11158, 9294, 8818, 8431, 8649, 2975, 1, 7228, 2074,
             3526, 2975, 10298, 11300, 951, 2975, 6750, 5746, 9477, 7228, 933, 7879, 10368, 3697,
             1149, 5739, 3927, 6765, 11723, 9772, 5765, 6765, 11395, 1176, 7228, 139, 5472, 1264,
             7895, 1875, 9430, 12105, 2975, 9703, 3697, 3824, 7895, 2001, 2582]]
    data_format = []
    for one in data:
        data_format.append([0] * (seq_length - len(one)) + one)
    return tf.constant(np.array(data_format))


label = [
    [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 2, 2, 2, 2, 1, 6, 6, 6, 6, 6, 6, 6, 6,
     6, 6, 3, 5, 4, 6],
    [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 5, 5, 5, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
     6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
     6, 6, 6, 6, 6, 6, 6, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
     6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [6, 3, 5, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 1, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]]
label_format = []
for one in label:
    label_format.append([6] * (seq_length - len(one)) + one)
# estimator内部机制会循环调用in_data，直到抛出异常（StopIteration）
results = classifier.predict(input_fn=lambda: in_data(), yield_single_examples=False)
for result in results:
    print(np.sum(result['seq_labels'] == label_format))  # 一共900个预测值，输出预测正确的个数
    break
