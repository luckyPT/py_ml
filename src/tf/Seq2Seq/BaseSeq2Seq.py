# coding:utf-8
import tensorflow as tf
import numpy as np

tf.enable_eager_execution()
batch_size = 32
vocabulary_size = 12149
embedding_size = 50
seq_length = 300
tag_size = 9  # 7表示 start，8表示pad

"""
此文件的测试场景：待编码的Seq和最后完成解码的Seq的长度是相等的
但实际上不相等应该也是可以的，可以通过增加特殊的结束标识来完成；
或者不使用padding，那么batch_size只能设置为1 来训练
"""


class Seq2Seq_RNN_RNN_Model(tf.keras.Model):
    """
    单层的RNN编码 + RNN解码；
    编码状态仅作用于解码的第一个时刻；
    可以改进的一个点是将编码最后的输出作为起始输入，而不是统一的使用start代替
    """

    def __init__(self):
        super(Seq2Seq_RNN_RNN_Model, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocabulary_size, embedding_size)
        self.encoder = tf.keras.layers.SimpleRNN(units=128, return_state=True)

        self.decoder_embedding = tf.keras.layers.Embedding(tag_size, embedding_size)
        self.decoder = tf.keras.layers.SimpleRNN(units=128, return_sequences=True)
        self.dense = tf.keras.layers.Dense(units=tag_size)

    def __call__(self, inputs, seq_label, is_train=True):
        last = self.encoder(self.embedding(inputs))
        last_output = last[0]
        state = last[1]
        if is_train:
            start = np.ones(shape=[batch_size, 1]) * 7
            decoder_input = tf.convert_to_tensor(np.concatenate((start, seq_label), axis=1)[:, :-1])
        else:
            decoder_input = tf.convert_to_tensor(seq_label)
        return self.dense(self.decoder(self.decoder_embedding(decoder_input), initial_state=state))


class Seq2Seq_2LSTM_2LSTM_Model(tf.keras.Model):
    """
    编码器&解码器均采用两层LSTM
    编码状态仅作用于解码的第一个时刻
    """

    def __init__(self):
        super(Seq2Seq_2LSTM_2LSTM_Model, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocabulary_size, embedding_size)
        self.encoder1 = tf.keras.layers.LSTM(units=128, return_sequences=True, return_state=True)
        self.encoder2 = tf.keras.layers.LSTM(units=128, return_state=True)

        self.decoder_embedding = tf.keras.layers.Embedding(tag_size, embedding_size)
        self.decoder1 = tf.keras.layers.LSTM(units=128, return_sequences=True)
        self.decoder2 = tf.keras.layers.LSTM(units=128, return_sequences=True)
        self.dense = tf.keras.layers.Dense(units=tag_size)

    def __call__(self, inputs, seq_label, is_train=True):
        embedded = self.embedding(inputs)
        encoder_lstm1 = self.encoder1(embedded)
        encoder_lstm2 = self.encoder2(encoder_lstm1[0])

        state1_h = encoder_lstm1[1]
        state1_c = encoder_lstm1[2]
        state2_h = encoder_lstm2[1]
        state2_c = encoder_lstm2[2]

        if is_train:
            start = np.ones(shape=[batch_size, 1]) * 7
            decoder_input = tf.convert_to_tensor(np.concatenate((start, seq_label), axis=1)[:, :-1])
        else:
            decoder_input = tf.convert_to_tensor(seq_label)

        decoder_embeded = self.decoder_embedding(decoder_input)
        decoder_lstm1 = self.decoder1(decoder_embeded, initial_state=[state1_h, state1_c])
        decoder_lstm2 = self.decoder2(decoder_lstm1, initial_state=[state2_h, state2_c])
        return self.dense(decoder_lstm2)


class Seq2Seq_2LSTM_2LSTM_Model2(tf.keras.Model):
    """
    编码器&解码器均采用两层LSTM
    编码的状态作为解码每一个时刻的输入
    """

    def __init__(self):
        super(Seq2Seq_2LSTM_2LSTM_Model2, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocabulary_size, embedding_size)
        self.encoder1 = tf.keras.layers.LSTM(units=128, return_sequences=True, return_state=True)
        self.encoder2 = tf.keras.layers.LSTM(units=128, return_state=True)

        self.decoder_embedding = tf.keras.layers.Embedding(tag_size, embedding_size)
        self.decoder1 = tf.keras.layers.LSTM(units=128, return_sequences=True)
        self.decoder2 = tf.keras.layers.LSTM(units=128, return_sequences=True)
        self.dense = tf.keras.layers.Dense(units=tag_size)

    def __call__(self, inputs, seq_label, is_train=True):
        embedded = self.embedding(inputs)
        encoder_lstm1 = self.encoder1(embedded)
        encoder_lstm2 = self.encoder2(encoder_lstm1[0])

        state1_h = encoder_lstm1[1]
        state1_c = encoder_lstm1[2]
        state2_h = encoder_lstm2[1]
        state2_c = encoder_lstm2[2]

        if is_train:
            start = np.ones(shape=[batch_size, 1]) * 7
            decoder_input = tf.convert_to_tensor(np.concatenate((start, seq_label), axis=1)[:, :-1])
        else:
            decoder_input = tf.convert_to_tensor(seq_label)

        decoder_embeded = self.decoder_embedding(decoder_input)
        concat_encode_state = tf.concat(
            (decoder_embeded,
             tf.keras.backend.repeat(state1_h, decoder_embeded.shape[1]),  # 先复制，然后再与每个时间步连接
             tf.keras.backend.repeat(state1_c, decoder_embeded.shape[1]),
             tf.keras.backend.repeat(state2_h, decoder_embeded.shape[1]),
             tf.keras.backend.repeat(state2_c, decoder_embeded.shape[1])), axis=2)
        decoder_lstm1 = self.decoder1(concat_encode_state)
        decoder_lstm2 = self.decoder2(decoder_lstm1)
        return self.dense(decoder_lstm2)


def get_data(file_path):
    with open(file_path) as file:
        seq1 = []
        seq2 = []
        for line in file:
            line = line.strip('\n')
            array = np.array(list(map(lambda x: x.split("/"), line.split("||"))), dtype=int)
            seq1.append(np.pad(array[:, 0], [0, seq_length - array.__len__()], mode="constant"))
            seq2.append(np.pad(array[:, 1], [0, seq_length - array.__len__()], mode="constant", constant_values=8))
            if len(seq1) >= batch_size:
                yield np.array(seq1), np.array(seq2)
                seq1 = []
                seq2 = []
        if len(seq1) < batch_size:
            print("get_data complete")
            raise StopIteration('complete')


def loss_function(real, preds):
    return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=preds)


data_iter = get_data("../../../data/ner/data.txt")

model = Seq2Seq_2LSTM_2LSTM_Model2()
EPOCHS = 10000
optimizer = tf.train.AdamOptimizer()
for epoch in range(EPOCHS):
    try:
        seq1, seq2 = data_iter.__next__()
        seq1 = tf.convert_to_tensor(seq1, dtype=tf.int32)
        with tf.GradientTape() as tape:
            predictions = model(seq1, seq2)
            loss = loss_function(seq2, predictions)
            print(loss)
            if epoch % 10 == 0:
                pre_lables = tf.argmax(predictions, axis=2)
                print(np.sum(np.sum(pre_lables.numpy() == seq2, axis=1) > 297))
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))
    except StopIteration as e:
        data_iter = get_data("../../../data/ner/data.txt")
        # Test
        seq1, seq2 = data_iter.__next__()
        seq1 = tf.convert_to_tensor(seq1, dtype=tf.int32)
        start_input = np.ones([batch_size, 1]) * 7
        pre_lables = None
        for i in range(seq_length):
            if pre_lables is None:
                input = start_input
            else:
                input = np.concatenate((start_input, pre_lables), axis=1)
            predictions = model(seq1, input, is_train=False)
            pre_lables = tf.argmax(predictions, axis=2)
        print(np.sum(pre_lables.numpy() == seq2, axis=1))
        print("one epoch complete... ...")