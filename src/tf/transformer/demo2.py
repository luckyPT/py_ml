import tensorflow as tf

import src.tf.transformer.mask as mask
import src.tf.transformer.train_data as train_data
from src.tf.transformer.transformer import Transformer

num_layers = 16
d_model = 128
dff = 512
num_heads = 8

dropout_rate = 0.1


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, loss_.dtype)
    mask = tf.reshape(mask, [-1, loss_.shape[1]])
    loss_ = loss_ * mask
    return tf.reduce_mean(loss_)


def train_data_iter():
    for (batch, (inp, tar)) in enumerate(train_dataset):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = mask.create_masks(inp, tar_inp)
        yield ((inp, tar_inp, enc_padding_mask, combined_mask, dec_padding_mask), tar_real)


input_vocab_size, target_vocab_size, train_dataset = train_data.get_data()

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/demo2/weights.{epoch:02d}.hdf5", verbose=1)

transformer = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate)
transformer.compile(optimizer=optimizer, loss=loss_function, metrics=["SparseCategoricalAccuracy"])

"""
# load weight之前需要先fit，用于初始化权重
transformer.fit_generator(train_data_iter(), steps_per_epoch=1000 // 32, epochs=1, callbacks=[checkpoint])
transformer.load_weights("./checkpoints/demo2/weights.60.hdf5")
"""
transformer.fit_generator(train_data_iter(), steps_per_epoch=1000 // 32, epochs=1000, callbacks=[checkpoint])
