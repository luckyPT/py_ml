import tensorflow as tf
import numpy as np

print(tf.__version__)


class GruCell:
    def __init__(self, units, step_dimension):
        """
        :param units: 每一个时间步的输出维度
        :param step_dimension: 每一时间步的输入维度
        """
        self.units = units
        self.en_w_r_z = tf.Variable(tf.truncated_normal(shape=[step_dimension + self.units, self.units * 2]))
        self.en_b_r_z = tf.Variable(tf.truncated_normal(shape=[units * 2, ]))
        self.en_w_h = tf.Variable(tf.truncated_normal(shape=[step_dimension + self.units, self.units]))
        self.en_b_h = tf.Variable(tf.truncated_normal(shape=[units, ]))

    def en_cond(self, i, en_embeded, en_gru_output):
        return i < tf.shape(en_embeded)[1]

    def en_gru(self, i, en_embeded, en_gru_output):
        step_in = en_embeded[:, i]
        last_state = en_gru_output[:, i]
        in_concat = tf.concat((step_in, last_state), axis=-1)
        gate_inputs = tf.sigmoid(tf.matmul(in_concat, self.en_w_r_z) + self.en_b_r_z)
        r, z = tf.split(value=gate_inputs, num_or_size_splits=2, axis=1)
        h_ = tf.tanh(tf.matmul(tf.concat((step_in, r * last_state), axis=-1), self.en_w_h) + self.en_b_h)
        h = z * last_state + (1 - z) * h_
        en_gru_output = tf.concat((en_gru_output, tf.expand_dims(h, axis=1)), axis=1)
        i = i + 1
        return i, en_embeded, en_gru_output

    def __call__(self, seqs, en_gru_output, *args, **kwargs):
        """
        在call函数内部创建en_gru_output会有问题，解码过程提示变量未初始化，所以在外部创建好变量传入call；估计可能是后面用这个变量的最后一个状态去初始化其他变量导致的
        :param seqs:
        :param en_gru_output:
        :param args:
        :param kwargs:
        :return:
        """
        i0 = tf.constant(0)
        _, _, encoder_output = tf.while_loop(self.en_cond, self.en_gru, loop_vars=[i0, seqs, en_gru_output],
                                             shape_invariants=[i0.get_shape(), seqs.get_shape(),
                                                               tf.TensorShape([None, None, self.units])])
        return encoder_output


class GruCellAttentionDecoder:
    def __init__(self, units, step_dimension):
        """
        :param units:每一个时间步的输出维度
        :param step_dimension: 每一时间步的输入维度
        """
        self.units = units
        self.de_w_r_z = tf.Variable(tf.truncated_normal(shape=[step_dimension + self.units * 2, self.units * 2]))
        self.de_b_r_z = tf.Variable(tf.truncated_normal(shape=[self.units * 2, ]))
        self.de_w_h = tf.Variable(tf.truncated_normal(shape=[step_dimension + self.units * 2, self.units]))
        self.de_b_h = tf.Variable(tf.truncated_normal(shape=[self.units, ]))

    def de_cond(self, i, de_embeded, de_gru_output, encoder_output):
        return i < tf.shape(de_embeded)[1]

    def de_gru(self, i, de_embeded, de_gru_output, encoder_output):
        step_in = de_embeded[:, i]
        last_state = de_gru_output[:, i]
        # 由于使用了tf.squeeze，所以在这里batch_size=1或者seq len=1时，可能会有问题
        attention_weight = tf.nn.softmax(tf.squeeze(tf.matmul(encoder_output, tf.expand_dims(last_state, axis=2))))
        context_c = tf.reduce_sum(tf.multiply(tf.expand_dims(attention_weight, axis=2), encoder_output), axis=1)
        step_in = tf.concat((step_in, context_c), axis=-1)
        in_concat = tf.concat((step_in, last_state), axis=-1)
        gate_inputs = tf.sigmoid(tf.matmul(in_concat, self.de_w_r_z) + self.de_b_r_z)
        r, z = tf.split(value=gate_inputs, num_or_size_splits=2, axis=1)
        h_ = tf.tanh(tf.matmul(tf.concat((step_in, r * last_state), axis=-1), self.de_w_h) + self.de_b_h)
        h = z * last_state + (1 - z) * h_
        de_gru_output = tf.concat((de_gru_output, tf.expand_dims(h, axis=1)), axis=1)
        i = i + 1
        return i, de_embeded, de_gru_output, encoder_output

    def __call__(self, de_embeded, de_gru_output, encoder_output, *args, **kwargs):
        """
        可以在内部创建de_gru_output，并不会抛出未初始化异常
        :param de_embeded:
        :param args:
        :param kwargs:
        :return:
        """
        i0 = tf.constant(0)
        _, _, decoder_output, _ = tf.while_loop(self.de_cond, self.de_gru,
                                                loop_vars=[i0, de_embeded, de_gru_output, encoder_output],
                                                shape_invariants=[i0.get_shape(), de_embeded.get_shape(),
                                                                  tf.TensorShape([None, None, self.units]),
                                                                  encoder_output.get_shape()])
        return decoder_output


input_data = np.array([[1, 2, 3, 4, 5], [2, 4, 1, 4, 5], [4, 3, 2, 3, 1]])  # batch_size=3,seq_length=5
input_label = np.array([[5, 4, 3, 2, 1], [4, 2, 5, 2, 1], [2, 3, 4, 3, 5]])

input_vocab_size = 6  # 从0开始，0可以认为padding
output_vocab_size = 7  # 需要考虑解码时的start，暂定6为start，0为padding
batch_size = 3
seq_len = 5
embedding_size = 4
gru_units = 4
sess = tf.Session()
# 编码
en_input = tf.placeholder(tf.int32, shape=[None, None])
en_embedding_variable = tf.Variable(tf.truncated_normal(shape=[input_vocab_size, embedding_size]))
en_embeded = tf.nn.embedding_lookup(en_embedding_variable, en_input)
en_gru_cell = GruCell(units=gru_units, step_dimension=embedding_size)
gru_init_state = tf.zeros(shape=[batch_size, 1, en_gru_cell.units])  # 这里不应该定义称为变量
encoder_output = en_gru_cell(en_embeded, gru_init_state)

# 解码
de_in_label = tf.placeholder(tf.int32, shape=[None, None])  # batch_size,seq_len
de_embedding_variable = tf.Variable(tf.truncated_normal(shape=[output_vocab_size, embedding_size]))
de_embeded = tf.nn.embedding_lookup(de_embedding_variable, de_in_label)
de_gru_cell = GruCellAttentionDecoder(gru_units, step_dimension=embedding_size)
de_init_state = tf.expand_dims(encoder_output[:, -1], axis=1)  # init state
decoder_output = de_gru_cell(de_embeded, de_init_state, encoder_output)
# 全连接
dense_w = tf.Variable(tf.truncated_normal(shape=[gru_units, output_vocab_size]))
dense_b = tf.Variable(tf.truncated_normal(shape=[output_vocab_size, ]))
output = tf.tensordot(decoder_output, dense_w, [[2], [0]]) + dense_b
# 易错点，labels=de_in_label[:, 1:] 应该从第1个元素开始，而不是第0个
loss = tf.losses.sparse_softmax_cross_entropy(labels=de_in_label[:, 1:], logits=output[:, 1:-1])

optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(loss)
decoder_start = np.zeros(shape=[batch_size, 1]) + output_vocab_size - 1
saver = tf.train.Saver()
# sess.run(tf.global_variables_initializer())
saver.restore(sess, "./test.model")
"""
for i in range(1000):
    #这里丢弃了input_label最后一个时间步，所以识别时不会识别最后一个时间步,实际应用中可以使用padding或者结束符来补上最后一个时间步
    print(sess.run((loss, optimizer),
                   feed_dict={en_input: input_data,
                              de_in_label: np.concatenate((decoder_start, input_label), axis=-1)[:, 0:-1]}))
saver.save(sess, save_path="./test.model")
"""
output = sess.run(output,
                  feed_dict={en_input: input_data,
                             de_in_label: np.concatenate((decoder_start, input_label), axis=-1)[:, 0:-1]})
print(np.argmax(output, axis=2)[:, 1:-1])
