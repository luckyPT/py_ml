# coding:utf-8
import tensorflow as tf
import numpy as np

from tensorflow.python.framework import graph_util


def to_pb(meta_path, model_path, output_file_path, node_names):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, model_path)
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, node_names)
        with tf.gfile.FastGFile(output_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())
            print('to pb success,file in ', output_file_path)


def load_pb():
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open("../../model/ner.pb", "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            graph = tf.get_default_graph()
            input_layer = graph.get_tensor_by_name("Placeholder:0")
            sequence_lengths_t = graph.get_tensor_by_name('PlaceholderWithDefault:0')
            out_seq = graph.get_tensor_by_name('ReverseSequence_1:0')
            seq_length = 300
            data = np.ones(shape=[1, 300])
            pre = sess.run(out_seq, feed_dict={input_layer: data,
                                               sequence_lengths_t: np.full(data.shape[0], seq_length, dtype=np.int32)})
            print(pre)


node_names = ['Placeholder', 'ReverseSequence_1', 'PlaceholderWithDefault']
to_pb("../../model/ner.model-84924.meta", "../../model/ner.model-84924", "../../model/ner.pb", node_names)
load_pb()
