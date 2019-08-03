import tensorflow as tf
import numpy as np

EMBEDDING_SIZE = 64
DIC_SIZE = 7148
LABEL_SIZE = 3440  # 0为padding，24为start
SEQ_LEN = 128
BATCH_SIZE = 512
GRU_UNITS = 256
model_path = "./"

feature_description = {
    "word_id": tf.io.VarLenFeature(dtype=tf.int64),
    "word_label": tf.io.VarLenFeature(dtype=tf.int64)
}


def parse_fn(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_description)
    f = tf.sparse.to_dense(example['word_id'])
    l = tf.sparse.to_dense(example['word_label'])
    return tf.pad(f, [[0, SEQ_LEN - tf.shape(f)[0]]]), tf.pad(l, [[0, SEQ_LEN - tf.shape(l)[0]]]), tf.shape(f)[0], \
           tf.shape(l)[0]


def input_fn(file_path):
    ds = tf.data.TFRecordDataset(file_path, num_parallel_reads=4) \
        .map(parse_fn, num_parallel_calls=4) \
        .shuffle(buffer_size=1024) \
        .batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(51200).repeat()
    return ds


class GruCell:
    def __init__(self, units, step_dimension):
        """
        :param units: 每一个时间步的输出维度
        :param step_dimension: 每一时间步的输入维度
        """
        self.units = units
        self.en_w_r_z = tf.Variable(tf.truncated_normal(shape=[step_dimension + self.units, self.units * 2]) / 10000)
        self.en_b_r_z = tf.Variable(tf.truncated_normal(shape=[units * 2, ]) / 10000)
        self.en_w_h = tf.Variable(tf.truncated_normal(shape=[step_dimension + self.units, self.units]) / 10000)
        self.en_b_h = tf.Variable(tf.truncated_normal(shape=[units, ]) / 10000)

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
        self.de_w_r_z = tf.Variable(
            tf.truncated_normal(shape=[step_dimension + self.units * 2, self.units * 2]) / 10000)
        self.de_b_r_z = tf.Variable(tf.truncated_normal(shape=[self.units * 2, ]) / 10000)
        self.de_w_h = tf.Variable(tf.truncated_normal(shape=[step_dimension + self.units * 2, self.units]) / 10000)
        self.de_b_h = tf.Variable(tf.truncated_normal(shape=[self.units, ]) / 10000)

    def de_cond(self, i, de_embeded, de_gru_output, encoder_output):
        return i < tf.shape(de_embeded)[1]

    def de_gru(self, i, de_embeded, de_gru_output, encoder_output):
        step_in = de_embeded[:, i]
        last_state = de_gru_output[:, i]
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


sess = tf.Session()
# 编码
en_input = tf.placeholder(tf.int32, shape=[None, None])
en_embedding_variable = tf.Variable(tf.truncated_normal(shape=[DIC_SIZE, EMBEDDING_SIZE]))
en_embeded = tf.nn.embedding_lookup(en_embedding_variable, en_input)
en_gru_cell_1 = GruCell(units=GRU_UNITS, step_dimension=EMBEDDING_SIZE)
gru_init_state_1 = tf.zeros(shape=[BATCH_SIZE, 1, en_gru_cell_1.units])  # 这里不应该定义称为变量
encoder_output_1 = en_gru_cell_1(en_embeded, gru_init_state_1)

en_gru_cell_2 = GruCell(units=GRU_UNITS, step_dimension=en_gru_cell_1.units)
gru_init_state_2 = tf.zeros(shape=[BATCH_SIZE, 1, en_gru_cell_2.units])  # 这里不应该定义称为变量
encoder_output_2 = en_gru_cell_2(encoder_output_1[:, 1:], gru_init_state_2)

# 解码
de_in_label = tf.placeholder(tf.int32, shape=[None, None])  # batch_size,seq_len
de_embedding_variable = tf.Variable(tf.truncated_normal(shape=[LABEL_SIZE, EMBEDDING_SIZE]))
de_embeded = tf.nn.embedding_lookup(de_embedding_variable, de_in_label)
de_gru_cell_1 = GruCellAttentionDecoder(GRU_UNITS, step_dimension=EMBEDDING_SIZE)
de_init_state_1 = tf.expand_dims(encoder_output_1[:, -1], axis=1)  # init state
decoder_output_1 = de_gru_cell_1(de_embeded, de_init_state_1, encoder_output_2[:, 1:])

de_gru_cell_2 = GruCell(GRU_UNITS, step_dimension=de_gru_cell_1.units)
de_init_state_2 = tf.expand_dims(encoder_output_2[:, -1], axis=1)  # init state
decoder_output_2 = de_gru_cell_2(decoder_output_1[:, 1:], de_init_state_2)

# 全连接
dense_w_1 = tf.Variable(tf.truncated_normal(shape=[GRU_UNITS, GRU_UNITS // 2]))
dense_b_1 = tf.Variable(tf.truncated_normal(shape=[GRU_UNITS // 2, ]))
dense_w_2 = tf.Variable(tf.truncated_normal(shape=[GRU_UNITS // 2, LABEL_SIZE]))
dense_b_2 = tf.Variable(tf.truncated_normal(shape=[LABEL_SIZE, ]))

dense_1 = tf.nn.leaky_relu(tf.tensordot(decoder_output_2, dense_w_1, [[2], [0]]) + dense_b_1)
output = tf.nn.leaky_relu(tf.tensordot(dense_1, dense_w_2, [[2], [0]]) + dense_b_2)

loss = tf.losses.sparse_softmax_cross_entropy(labels=de_in_label[:, 1:], logits=output[:, 1:-1])

optimizer = tf.train.AdamOptimizer(learning_rate=0.002).minimize(loss)
decoder_start = np.zeros(shape=[BATCH_SIZE, 1]) + LABEL_SIZE - 1

saver = tf.train.Saver()
ds = input_fn("../../../data/translate/train_data.tfrecord").make_one_shot_iterator().get_next()
# sess.run(tf.global_variables_initializer())


saver.restore(sess, save_path=model_path + "ner.model-1")


def save_model():
    np.savetxt("./params_v3/en_embeding", sess.run(en_embedding_variable))
    np.savetxt("./params_v3/de_embedding", sess.run(de_embedding_variable))
    np.savetxt("./params_v3/en_grucell_1_w_r_z", sess.run(en_gru_cell_1.en_w_r_z))
    np.savetxt("./params_v3/en_grucell_1_b_r_z", sess.run(en_gru_cell_1.en_b_r_z))
    np.savetxt("./params_v3/en_grucell_1_w_h", sess.run(en_gru_cell_1.en_w_h))
    np.savetxt("./params_v3/en_grucell_1_b_h", sess.run(en_gru_cell_1.en_b_h))
    # --
    np.savetxt("./params_v3/en_grucell_2_w_r_z", sess.run(en_gru_cell_2.en_w_r_z))
    np.savetxt("./params_v3/en_grucell_2_b_r_z", sess.run(en_gru_cell_2.en_b_r_z))
    np.savetxt("./params_v3/en_grucell_2_w_h", sess.run(en_gru_cell_2.en_w_h))
    np.savetxt("./params_v3/en_grucell_2_b_h", sess.run(en_gru_cell_2.en_b_h))
    # --
    np.savetxt("./params_v3/de_grucell_1_w_r_z", sess.run(de_gru_cell_1.de_w_r_z))
    np.savetxt("./params_v3/de_grucell_1_b_r_z", sess.run(de_gru_cell_1.de_b_r_z))
    np.savetxt("./params_v3/de_grucell_1_w_h", sess.run(de_gru_cell_1.de_w_h))
    np.savetxt("./params_v3/de_grucell_1_b_h", sess.run(de_gru_cell_1.de_b_h))

    np.savetxt("./params_v3/de_grucell_2_w_r_z", sess.run(de_gru_cell_2.en_w_r_z))
    np.savetxt("./params_v3/de_grucell_2_b_r_z", sess.run(de_gru_cell_2.en_b_r_z))
    np.savetxt("./params_v3/de_grucell_2_w_h", sess.run(de_gru_cell_2.en_w_h))
    np.savetxt("./params_v3/de_grucell_2_b_h", sess.run(de_gru_cell_2.en_b_h))

    np.savetxt("./params_v3/dense_w_1", sess.run(dense_w_1))
    np.savetxt("./params_v3/dense_b_1", sess.run(dense_b_1))

    np.savetxt("./params_v3/dense_w_2", sess.run(dense_w_2))
    np.savetxt("./params_v3/dense_b_2", sess.run(dense_b_2))


min_v = 0.1
for i in range(100000):
    features, labels, f_lengths, l_lengths = sess.run(ds)
    """
    这里在训练的时候不同批次的最大长度是不一样的，目的是为了加快训练速度，但对精度会有影响，
    通过翻译场景来看，在预测时，适当的增加padding，对最终效果有一定的正面影响;
    后续需要优化的地方是重新组织训练数据，比如长度小于10的分到一个批次，padding长度为10； 10 ～ 20的分到一个批次，padding长度为20
    如果使用固定长度的padding训练，那么对算力的需求可能会增长几十倍
    """
    f_max_len = f_lengths[np.argmax(f_lengths, axis=-1)]
    l_max_len = l_lengths[np.argmax(l_lengths, axis=-1)]

    loss_value, _ = sess.run((loss, optimizer), feed_dict={en_input: features[:, :f_max_len + 1],
                                                           de_in_label: np.concatenate(
                                                               (decoder_start, labels[:, :l_max_len + 1]),
                                                               axis=-1)[:,
                                                                        0:-1]})

    print(i, loss_value)
    if i % 10 == 0 and (min_v > loss_value):
        saver.save(sess, model_path + "ner.model", global_step=int(loss_value * 1000))
        min_v = loss_value

    """
    if i % 1 == 0:
        # save_model()
        # saver.save(sess, model_path + "ner.model", global_step=int(loss_value * 1000))

        pred = sess.run(output,
                        feed_dict={en_input: features[:, :f_max_len + 1],
                                   de_in_label: np.concatenate((decoder_start, labels[:, :l_max_len + 1]),
                                                               axis=-1)[:, 0:-1]})
        print(i, np.sum(np.sum(np.argmax(pred, axis=2)[:, 1:-1] == labels[:, :l_max_len], axis=1) == l_max_len))
    """
sess.close()
