import numpy as np
import tensorflow as tf

"""
by panTeng
date 2019.7.21
"""
print(tf.__version__)
input_data = np.array([[1, 2, 3, 4, 5], [2, 4, 1, 4, 5], [4, 3, 2, 3, 1]])  # batch_size=3,seq_length=5
input_label = np.array([[5, 4, 3, 2, 1], [4, 2, 5, 2, 1], [2, 3, 4, 3, 5]])

input_vocab_size = 6  # 从0开始，0可以认为padding
output_vocab_size = 7  # 需要考虑解码时的start，暂定6为start，0为padding
batch_size = 3
seq_len = 5
embedding_size = 4
gru_units = 4

# 编码逻辑
en_input = tf.placeholder(tf.int32, shape=[None, None])
en_embedding_variable = tf.Variable(tf.truncated_normal(shape=[input_vocab_size, embedding_size]))
en_embeded = tf.nn.embedding_lookup(en_embedding_variable, en_input)
en_gru_init_state = tf.zeros(shape=[batch_size, 1, gru_units])  # 这里不应该定义称为变量
en_w_r_z = tf.Variable(tf.truncated_normal(shape=[embedding_size + gru_units, gru_units * 2]))
en_b_r_z = tf.Variable(tf.truncated_normal(shape=[gru_units * 2, ]))
en_w_h = tf.Variable(tf.truncated_normal(shape=[embedding_size + gru_units, gru_units]))
en_b_h = tf.Variable(tf.truncated_normal(shape=[gru_units, ]))


def en_cond(i, en_embeded, en_gru_output):
    return i < tf.shape(en_embeded)[1]


def en_gru(i, en_embeded, en_gru_output):
    step_in = en_embeded[:, i]
    last_state = en_gru_output[:, i]
    in_concat = tf.concat((step_in, last_state), axis=-1)
    gate_inputs = tf.sigmoid(tf.matmul(in_concat, en_w_r_z) + en_b_r_z)
    r, z = tf.split(value=gate_inputs, num_or_size_splits=2, axis=1)
    h_ = tf.tanh(tf.matmul(tf.concat((step_in, r * last_state), axis=-1), en_w_h) + en_b_h)
    h = z * last_state + (1 - z) * h_
    en_gru_output = tf.concat((en_gru_output, tf.expand_dims(h, axis=1)), axis=1)
    i = i + 1
    return i, en_embeded, en_gru_output


i0 = tf.constant(0)
_, _, encoder_output = tf.while_loop(en_cond, en_gru, loop_vars=[i0, en_embeded, en_gru_init_state],
                                     shape_invariants=[i0.get_shape(), en_embeded.get_shape(),
                                                       tf.TensorShape([batch_size, None, gru_units])])
# 解码逻辑
de_in_label = tf.placeholder(tf.int32, shape=[None, None])  # batch_size,seq_len
de_embedding_variable = tf.Variable(tf.truncated_normal(shape=[output_vocab_size, embedding_size]))
de_embeded = tf.nn.embedding_lookup(de_embedding_variable, de_in_label)
de_init_state = tf.expand_dims(encoder_output[:, -1], axis=1)  # init_state
de_w_r_z = tf.Variable(tf.truncated_normal(shape=[embedding_size + gru_units * 2, gru_units * 2]))
de_b_r_z = tf.Variable(tf.truncated_normal(shape=[gru_units * 2, ]))
de_w_h = tf.Variable(tf.truncated_normal(shape=[embedding_size + gru_units * 2, gru_units]))
de_b_h = tf.Variable(tf.truncated_normal(shape=[gru_units, ]))


def de_cond(i, de_embeded, de_gru_output):
    return i < tf.shape(de_embeded)[1]


def de_gru(i, de_embeded, de_gru_output):
    step_in = de_embeded[:, i]
    last_state = de_gru_output[:, i]
    attention_weight = tf.nn.softmax(tf.matmul(encoder_output, tf.expand_dims(last_state, axis=2)))
    context_c = tf.reduce_sum(tf.multiply(attention_weight, encoder_output), axis=1)
    step_in = tf.concat((step_in, context_c), axis=-1)
    in_concat = tf.concat((step_in, last_state), axis=-1)
    gate_inputs = tf.sigmoid(tf.matmul(in_concat, de_w_r_z) + de_b_r_z)
    r, z = tf.split(value=gate_inputs, num_or_size_splits=2, axis=1)
    h_ = tf.tanh(tf.matmul(tf.concat((step_in, r * last_state), axis=-1), de_w_h) + de_b_h)
    h = z * last_state + (1 - z) * h_
    de_gru_output = tf.concat((de_gru_output, tf.expand_dims(h, axis=1)), axis=1)
    i = i + 1
    return i, de_embeded, de_gru_output


i0 = tf.constant(0)
_, _, decoder_output = tf.while_loop(de_cond, de_gru, loop_vars=[i0, de_embeded, de_init_state],
                                     shape_invariants=[i0.get_shape(), de_embeded.get_shape(),
                                                       tf.TensorShape([batch_size, None, gru_units])])

# 全连接
dense_w = tf.Variable(tf.truncated_normal(shape=[gru_units, output_vocab_size]))
dense_b = tf.Variable(tf.truncated_normal(shape=[output_vocab_size, ]))
output = tf.tensordot(decoder_output, dense_w, [[2], [0]]) + dense_b
loss = tf.losses.sparse_softmax_cross_entropy(labels=input_label, logits=output[:, 1:])

optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(loss)
decoder_start = np.zeros(shape=[batch_size, 1]) + output_vocab_size - 1
with tf.Session() as sess:
    sess.run((tf.local_variables_initializer(), tf.global_variables_initializer()))
    for i in range(1000):
        print(sess.run((loss, optimizer),
                       feed_dict={en_input: input_data,
                                  de_in_label: np.concatenate((decoder_start, input_label), axis=-1)[:, 0:-1]}))
    output = sess.run(output,
                      feed_dict={en_input: input_data,
                                 de_in_label: np.concatenate((decoder_start, input_label), axis=-1)[:, 0:-1]})
    # print(output)
    print(np.argmax(output, axis=2)[:, 1:])
