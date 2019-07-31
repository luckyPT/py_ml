import math

import numpy as np

en_embeding = np.loadtxt("./params_v3/en_embeding")
de_embeding = np.loadtxt("./params_v3/de_embedding")
en_gru_w_r_z = np.loadtxt("./params_v3/en_grucell_w_r_z")
en_gru_b_r_z = np.loadtxt("./params_v3/en_grucell_b_r_z")
en_gru_w_h = np.loadtxt("./params_v3/en_grucell_w_h")
en_gru_b_h = np.loadtxt("./params_v3/en_grucell_b_h")
de_gru_w_r_z = np.loadtxt("./params_v3/de_grucell_w_r_z")
de_gru_b_r_z = np.loadtxt("./params_v3/de_grucell_b_r_z")
de_gru_w_h = np.loadtxt("./params_v3/de_grucell_w_h")
de_gru_b_h = np.loadtxt("./params_v3/de_grucell_b_h")
dense_w = np.loadtxt("./params_v3/dense_w")
dense_b = np.loadtxt("./params_v3/dense_b")


def sigmoid(x):
    return 1 / (1 + np.power(math.e, -1 * x))


def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


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

    def __call__(self, seqs, *args, **kwargs):
        seq_len = seqs.shape[0]
        last_state = np.zeros(shape=(self.units,))
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


en_gru_cell = GruCell(en_gru_w_r_z, en_gru_b_r_z, en_gru_w_h, en_gru_b_h)
decoder = GruCellAttentionDecoder(de_gru_w_r_z, de_gru_b_r_z, de_gru_w_h, de_gru_b_h)


def predict(ids):
    en_embeded = en_embeding[ids]
    encoder_vec = en_gru_cell(en_embeded)
    init_state = encoder_vec[-1]
    firs_in = de_embeding[6]
    ret_ids = []
    for i in range(len(ids[0])):
        state = decoder(firs_in, init_state, encoder_vec)
        out = np.dot(np.reshape(state, (1, 4)), dense_w) + dense_b
        firs_in = np.argmax(out, -1)
        ret_ids.append(firs_in[0])
        firs_in = de_embeding[firs_in[0]]
        init_state = state
    return ret_ids


if __name__ == '__main__':
    decoder_out = predict([[2, 4, 1, 4, 5]])
    print(decoder_out)
