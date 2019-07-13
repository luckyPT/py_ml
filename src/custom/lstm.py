import tensorflow as tf
import numpy as np
import math


# tf.enable_eager_execution()


def sigmoid(x):
    return 1 / (1 + np.power(math.e, -1 * x))


lstm_units = 2


class BaseLSTM(tf.keras.Model):
    def __init__(self):
        super(BaseLSTM, self).__init__()
        self.lstm = tf.keras.layers.LSTM(lstm_units, recurrent_activation='sigmoid', return_sequences=True)

    def call(self, input, training=True):
        return self.lstm(input)

    def save(self):
        np.savetxt("./lstm_kernel", self.lstm.cell.kernel.numpy())
        np.savetxt("./lstm_recurrent_kernel", self.lstm.cell.recurrent_kernel.numpy())
        np.savetxt("./lstm_bias", self.lstm.cell.bias.numpy())


class CustomLSTM:
    def __init__(self, kernel, r_kernel, bias):
        self.kernel = np.loadtxt("./lstm_kernel")  # kernel
        self.r_kernel = np.loadtxt("./lstm_recurrent_kernel")  # r_kernel
        self.bias = np.loadtxt("./lstm_bias")  # bias
        self.units = lstm_units
        # print(self.kernel)
        # print(self.r_kernel)
        # print(self.bias)

    def __call__(self, input):
        fw = []
        h_zero_state = np.zeros(shape=[1, self.units], dtype=float)
        c_zero_state = np.zeros(shape=[1, self.units], dtype=float)
        for index in range(seqs.shape[0]):
            x_input = np.matmul(seqs[index], self.kernel) + self.bias[:self.units * 4]
            x_i, x_f, x_c, x_o = np.split(x_input, indices_or_sections=4, axis=0)

            h_input = np.matmul(h_zero_state, self.r_kernel)  # + self.bias[self.units * 4:]
            h_i, h_f, h_c, h_o = np.split(h_input, indices_or_sections=4, axis=1)

            i = sigmoid(x_i + h_i)
            f = sigmoid(x_f + h_f)
            o = sigmoid(x_o + h_o)
            c_tmp = np.tanh(x_c + h_c)
            c = f * c_zero_state + i * c_tmp
            h = o * np.tanh(c)
            fw.append(h)
            h_zero_state = h
            c_zero_state = c
        return fw


if __name__ == '__main__':
    seqs = tf.constant([[[1.2, 1.1], [1.01, 1.05], [1, 1], [1, 1], [1, 1], [1.05, 1.12]]], dtype=float)

    model_path = "./tmp_model_weight.h5"
    model = BaseLSTM()
    model(seqs)  # 必须先加载一次，才能load_weights
    model.save_weights(model_path)
    model.save()
    model.load_weights(model_path)
    print(model(seqs))

    # custom
    seqs = np.array([[1.2, 1.1], [1.01, 1.05], [1, 1], [1, 1], [1, 1], [1.05, 1.12]])

    model = CustomLSTM(None, None, None)
    print(model(seqs))
