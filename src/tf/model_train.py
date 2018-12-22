# coding:utf-8
import numpy as np
import tensorflow as tf

batch_size = 4
seq_length = 300
vocabulary_size = 12149
embedding_size = 50
print(tf.__version__)


def get_data(file_path, batch_size):
    with open(file_path) as file:
        batch_words = []
        batch_labels = []
        for line in file:
            words_label = list(map(lambda x: x.split("/"), line.replace("\n", "").split("||")))
            words_label = np.array(words_label)
            words = words_label[:, 0]
            labels = words_label[:, 1]
            words = np.pad(words, (seq_length - words.shape[0], 0), mode="constant", constant_values=0)
            labels = np.pad(labels, (seq_length - labels.shape[0], 0), mode="constant", constant_values=0)
            batch_words.append(words)
            batch_labels.append(labels)
            if len(batch_labels) == batch_size:
                yield np.array(batch_words), np.array(batch_labels)
                batch_words.clear()
                batch_labels.clear()


embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

input_layer = tf.placeholder(tf.int32, [None, seq_length])
labels = tf.placeholder(tf.int32, [None, seq_length])
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
sequence_lengths_t = tf.placeholder_with_default(np.full(batch_size, seq_length, dtype=np.int32), [None, ])
log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(dense, labels, sequence_lengths_t)
loss = tf.reduce_mean(-log_likelihood)
tf.summary.scalar('mean', loss)
train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(
    dense, transition_params, sequence_lengths_t)
merged = tf.summary.merge_all()
saver = tf.train.Saver()

data = get_data("../../data/ner/data.txt", batch_size)
with tf.Session() as sess:
    # sess.run((tf.local_variables_initializer(), tf.global_variables_initializer()))
    saver = tf.train.import_meta_graph("../../model/ner.model-106195.meta")
    saver.restore(sess, "../../model/ner.model-106195")
    train_writer = tf.summary.FileWriter("../../log", sess.graph)
    # 训练
    for i in range(20):
        feature, label = data.__next__()
        _, cost, summary = sess.run((train_op, loss, merged),
                                    feed_dict={
                                        input_layer: np.reshape(feature, ([batch_size, seq_length])),
                                        labels: np.reshape(label, ([batch_size, seq_length]))})
        train_writer.add_summary(summary, i)
        print(cost)

    # 测试
    test_feature, test_label = data.__next__()
    tf_viterbi_sequence = sess.run(viterbi_sequence,
                                   feed_dict={
                                       input_layer: test_feature.reshape([batch_size, seq_length])})
    print(tf_viterbi_sequence)

    # 保存模型
    saver.save(sess, "../../model/ner.model", global_step=int(cost * 1000))
