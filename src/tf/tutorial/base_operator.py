import tensorflow as tf
import numpy as np

print(tf.__version__)


def def_constant():
    """
    常量声明
    :return:
    """
    # 标量
    a = tf.constant(1)
    # 一维张量
    a1 = tf.constant([2])
    # 二维张量
    a2 = tf.constant([[1, 2, 3], [4, 5, 6]])
    # 三维张量
    a3 = tf.constant([[[1, 2], [1, 2]], [[4, 5], [4, 5]]])
    print("0维：" + str(sess.run(a)))
    print("1维：" + str(sess.run(a1)))
    print("2维：" + str(sess.run(a2)))
    print("3维：" + str(sess.run(a3)))


def def_var():
    """
    变量声明：
    变量声明的两种方式：tf.Variable & tf.get_variable
    前者是调用构造方法创建一个新的对象，并且不能被复用
    后者通过设置reuse=tf.AUTO_REUSE，优先看当前scope下面是否有符合要求的变量，有则返回这个值，没有则创建新的对象；
    设置为tf.reuse=True,会抛异常，可能与版本有关系
    :return:
    """

    # 默认会放在计算最快速的设备上（比如GPU），可以通过with tf.device("/device:GPU:1"):来修改计算位置
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        my_var = tf.Variable([[1., 2., 3.], [4., 5., 6.]], name="my_var")
        # tf.get_variable 原理是先检查当前命名空间下有没有这个变量，如果有则返回这个变量，没有则创建
        var2 = tf.get_variable("my_var", shape=[2, 3], initializer=tf.constant_initializer([[1, 2, 3], [4, 5, 6]]))
        var3 = tf.get_variable("my_var", shape=[2, 3], initializer=tf.constant_initializer([[2, 2, 2], [3, 3, 3]]))

        sess.run(tf.initialize_all_variables())  # 使用之前应该先初始化
        print("my_var==var2:" + str(my_var == var2) + "  var2==var3:" + str(var2 == var3))  # False True
        print("my_var:" + str(sess.run(my_var)))
        print("var2" + str(sess.run(var2)))
        print("var3" + str(sess.run(var3)))


def add_sub_multiply_div_emlement_wise():
    """
    对应元素的加减乘除
    :return:
    """
    a2 = tf.constant([[1, 2, 3], [4, 5, 6]])
    # 加法
    a2_add_a2 = tf.add(a2, a2)
    print("a2+a2=" + str(sess.run(a2_add_a2)))
    # 减法
    a2_sub_a2 = tf.subtract(a2, a2)
    print("a2-a2=" + str(sess.run(a2_sub_a2)))
    # 乘法
    a2_multiply_a2 = tf.multiply(a2, a2)
    print("a2*a2=" + str(sess.run(a2_multiply_a2)))
    # 除法
    a2_div_a2 = tf.div(a2, a2)
    print("a2/a2=" + str(sess.run(a2_div_a2)))


def metrix_opration():
    """
    常见的矩阵操作
    :return:
    """
    x = tf.constant([[[1, 2, 3],
                      [4, 5, 6]],
                     [[7, 8, 9],
                      [10, 11, 12]]])
    x_t_default = tf.transpose(x)
    print("x_t_default=" + str(sess.run(x_t_default)))
    x_t021 = tf.transpose(x, perm=[0, 2, 1])
    print("x_t=" + str(sess.run(x_t021)))
    a2 = tf.constant([[1, 2, 3], [4, 5, 6]])
    # 转置运算
    a2_t = tf.transpose(a2)
    print("转置：" + str(sess.run(a2_t)))
    # 点积与转置运算
    a2_dot_a2_t = tf.matmul(a2, a2, transpose_b=True)
    print("a2·a2=" + str(sess.run(a2_dot_a2_t)))
    # 维度变换(大致思路是先flattern，然后再依次构造)
    a2_reshape213 = tf.reshape(a2, [2, 1, 3])
    a2_reshape231 = tf.reshape(a2, [2, 3, 1])
    a2_reshape321 = tf.reshape(a2, [3, 2, 1])
    a2_reshape16 = tf.reshape(a2, [1, 6])
    a2_reshape6_1 = tf.reshape(a2, [6, -1])
    a2_reshape_flattern = tf.reshape(a2, [-1])
    print("a2_reshape213=" + str(sess.run(a2_reshape213)))
    print("a2_reshape231=" + str(sess.run(a2_reshape231)))
    print("a2_reshape321=" + str(sess.run(a2_reshape321)))
    print("a2_reshape16=" + str(sess.run(a2_reshape16)))
    print("a2_reshape6_1=" + str(sess.run(a2_reshape6_1)))
    print("a2_reshape_flattern=" + str(sess.run(a2_reshape_flattern)))
    # 一维tensor转标量
    scalar_tensor = tf.constant([[1]])
    tensor2scalar = tf.reshape(scalar_tensor, [])
    print("tensor2scala=" + str(sess.run(tensor2scalar)))
    # 扩展维度与缩减维度
    a2_expand1 = tf.expand_dims(a2, 1)  # [2,3]->[2,1,3]
    print("扩展维度1：" + str(sess.run(a2_expand1)))
    a2_expand_1 = tf.expand_dims(a2, -1)  # [2,3]->[2,3,1]
    print("扩展维度1：" + str(sess.run(a2_expand_1)))
    remove_all_dim1 = tf.squeeze(a2_expand_1)  # 可以通过axis指定需要删除哪些维度，axis可以是一个int型列表
    print("缩减维度1：" + str(sess.run(remove_all_dim1)))  # [2,3,1]->[2,3]


def other_common_operation():
    """
    tf.random.normal\tf.reduce_sum\tf.convert_to_tensor
    :return:
    """
    # random.normal 正太分布
    normal_var = tf.random.normal(shape=[3, 4, 5])
    print("normal_var = " + str(sess.run(normal_var)))
    # reduce
    reduce_normal = tf.reduce_sum(normal_var, axis=0)  # 0->[4,5] 1->[3,5] 2->[3,4]
    print("after reduce shape:" + str(reduce_normal.shape))
    print(sess.run(reduce_normal))
    # tf.convert_to_tensor 转tensor
    A = list([1, 2, 3])
    B = np.array([1, 2, 3])
    C = tf.convert_to_tensor(A)
    D = tf.convert_to_tensor(B)
    print(type(A), type(B), type(C), type(D), sep="\n")
    print("C=" + str(sess.run(C)), "D=" + str(sess.run(D)), sep="\n")


if __name__ == '__main__':
    sess = tf.Session()
    # 常量的声明并求值
    # def_constant()
    # def_var()
    # add_sub_multiply_div_emlement_wise()
    # metrix_opration()
    other_common_operation()
    sess.close()
