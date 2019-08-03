import numpy as np
import tensorflow as tf
import math

en_embeding = np.loadtxt("./params_v3/en_embeding")
de_embeding = np.loadtxt("./params_v3/de_embedding")
en_gru_1_w_r_z = np.loadtxt("./params_v3/en_grucell_1_w_r_z")
en_gru_1_b_r_z = np.loadtxt("./params_v3/en_grucell_1_b_r_z")
en_gru_1_w_h = np.loadtxt("./params_v3/en_grucell_1_w_h")
en_gru_1_b_h = np.loadtxt("./params_v3/en_grucell_1_b_h")
en_gru_2_w_r_z = np.loadtxt("./params_v3/en_grucell_2_w_r_z")
en_gru_2_b_r_z = np.loadtxt("./params_v3/en_grucell_2_b_r_z")
en_gru_2_w_h = np.loadtxt("./params_v3/en_grucell_2_w_h")
en_gru_2_b_h = np.loadtxt("./params_v3/en_grucell_2_b_h")
de_gru_1_w_r_z = np.loadtxt("./params_v3/de_grucell_1_w_r_z")
de_gru_1_b_r_z = np.loadtxt("./params_v3/de_grucell_1_b_r_z")
de_gru_1_w_h = np.loadtxt("./params_v3/de_grucell_1_w_h")
de_gru_1_b_h = np.loadtxt("./params_v3/de_grucell_1_b_h")
de_gru_2_w_r_z = np.loadtxt("./params_v3/de_grucell_2_w_r_z")
de_gru_2_b_r_z = np.loadtxt("./params_v3/de_grucell_2_b_r_z")
de_gru_2_w_h = np.loadtxt("./params_v3/de_grucell_2_w_h")
de_gru_2_b_h = np.loadtxt("./params_v3/de_grucell_2_b_h")
dense_w_1 = np.loadtxt("./params_v3/dense_w_1")
dense_b_1 = np.loadtxt("./params_v3/dense_b_1")
dense_w_2 = np.loadtxt("./params_v3/dense_w_2")
dense_b_2 = np.loadtxt("./params_v3/dense_b_2")


def sigmoid(x):
    return 1 / (1 + np.power(math.e, -1 * x))


def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def leaky_relu(x):
    return np.where(x > 0, x, x * 0.2)


class GruCell:
    def __init__(self, w_r_z, b_r_z, w_h, b_h):
        """
        :param units: 每一个时间步的输出维度
        :param step_dimension: 每一时间步的输入维度
        """
        self.w_r_z = w_r_z
        self.b_r_z = b_r_z
        self.w_h = w_h
        self.b_h = b_h
        self.units = self.w_h.shape[1]
        self.step_d = self.w_h.shape[0] - self.w_h.shape[1]

    def __call__(self, seqs, last_state=None, *args, **kwargs):
        if last_state is None:
            last_state = np.zeros(shape=(self.units,))
        if np.ndim(seqs) == 1:
            seqs = np.expand_dims(seqs, axis=0)

        seq_len = seqs.shape[0]
        en_gru_output = [last_state]
        for i in range(seq_len):
            step_in = seqs[i]
            in_concat = np.concatenate((step_in, last_state), axis=-1)
            gate_inputs = sigmoid(np.matmul(in_concat, self.w_r_z) + self.b_r_z)
            r, z = np.split(gate_inputs, 2, axis=0)
            h_ = np.tanh(np.matmul(np.concatenate((step_in, r * last_state), axis=-1), self.w_h) + self.b_h)
            h = z * last_state + (1 - z) * h_
            en_gru_output.append(h)
            last_state = h
        return np.array(en_gru_output)


class GruCellAttentionDecoder():
    def __init__(self, w_r_z, b_r_z, w_h, b_h):
        self.w_r_z = w_r_z
        self.b_r_z = b_r_z
        self.w_h = w_h
        self.b_h = b_h
        self.units = self.w_h.shape[1]
        self.step_d = self.w_h.shape[0] - self.w_h.shape[1]

    def __call__(self, step_in, last_state, encoder_output, *args, **kwargs):
        attention_weight = softmax(np.matmul(encoder_output, last_state))
        context_c = np.sum(np.multiply(np.expand_dims(attention_weight, axis=1), encoder_output), axis=0)
        step_in = np.concatenate((step_in, context_c), axis=-1)
        in_concat = np.concatenate((step_in, last_state), axis=-1)
        gate_inputs = sigmoid(np.matmul(in_concat, self.w_r_z) + self.b_r_z)
        r, z = np.split(gate_inputs, 2, axis=0)
        h_ = np.tanh(np.matmul(np.concatenate((step_in, r * last_state), axis=-1), self.w_h) + self.b_h)
        h = z * last_state + (1 - z) * h_
        return h


en_gru_cell_1 = GruCell(en_gru_1_w_r_z, en_gru_1_b_r_z, en_gru_1_w_h, en_gru_1_b_h)
en_gru_cell_2 = GruCell(en_gru_2_w_r_z, en_gru_2_b_r_z, en_gru_2_w_h, en_gru_2_b_h)
decoder_1 = GruCellAttentionDecoder(de_gru_1_w_r_z, de_gru_1_b_r_z, de_gru_1_w_h, de_gru_1_b_h)
decoder_2 = GruCell(de_gru_2_w_r_z, de_gru_2_b_r_z, de_gru_2_w_h, de_gru_2_b_h)


def predict(ids, decoder_len=None):
    if decoder_len is None:
        decoder_len = len(ids[0])
    en_embeded = en_embeding[ids]
    encoder_vec1 = en_gru_cell_1(en_embeded)
    encoder_vec2 = en_gru_cell_2(encoder_vec1[1:])
    init_state_1 = encoder_vec1[-1]
    init_state_2 = encoder_vec2[-1]
    firs_in = de_embeding[3439]
    ret_ids = []
    for i in range(decoder_len):
        state_1 = decoder_1(firs_in, init_state_1, encoder_vec2[1:])
        state_2 = decoder_2(state_1, init_state_2)[1]
        dense_1 = leaky_relu(np.dot(np.reshape(state_2, (1, decoder_2.units)), dense_w_1) + dense_b_1)
        dense_2 = leaky_relu(np.dot(np.reshape(dense_1, (1, dense_1.size)), dense_w_2) + dense_b_2)
        firs_in = np.argmax(dense_2, -1)[0]
        ret_ids.append(firs_in)
        firs_in = de_embeding[firs_in]
        init_state_1 = state_1
        init_state_2 = state_2
    return ret_ids


if __name__ == '__main__':
    with open("/home/panteng/PycharmProjects/sms-ner-py/data/translate/train_data.txt") as file:
        t_count = 0
        f_count = 0
        for line in file:
            f, l = line.strip("\n").split("\t")
            f = list(map(lambda x: int(x), f.split(",")))
            for i in range(10):
                f.append(0)
            l = list(map(lambda x: int(x), l.split(",")))
            for i in range(10):
                l.append(0)

            pre_seqs = predict([f], decoder_len=len(l))
            if pre_seqs == l:
                t_count += 1
            else:
                f_count += 1
            if t_count % 1000 == 0:
                print(t_count, f_count)
        print(t_count, f_count)
