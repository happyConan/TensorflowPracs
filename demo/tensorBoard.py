#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 创建一个神经网络层
def add_layer(input, in_size, out_size, activation_function=None):
    """
    :param input:
        神经网络层的输入
    :param in_zize:
        输入数据的大小
    :param out_size:
        输出数据的大小
    :param activation_function:
        神经网络激活函数，默认没有
    """
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            # 定义神经网络的初始化权重
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        with tf.name_scope('biases'):
            # 定义神经网络的偏置
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        with tf.name_scope('W_mul_x_plus_b'):
            # 计算w*x+b
            W_mul_x_plus_b = tf.matmul(input, Weights) + biases
        # 根据是否有激活函数进行处理
        if activation_function is None:
            output = W_mul_x_plus_b
        else:
            output = activation_function(W_mul_x_plus_b)

        return output


# 创建一个具有输入层、隐藏层、输出层的三层神经网络，神经元个数分别为1,10,1
# 创建只有一个特征的输入数据，数据数目为300，输入层
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# 创建数据中的噪声
noise = np.random.normal(0, 0.05, x_data.shape)
# 创建输入数据对应的输出
y_data = np.square(x_data) + 1 + noise

with tf.name_scope('input'):
    # 定义输入数据，None是样本数目，表示多少输入数据都行，1是输入数据的特征数目
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    # 定义输出数据，与xs同理
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# 定义一个隐藏层
hidden_layer = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# 定义输出层
prediction = add_layer(hidden_layer, 10, 1, activation_function=None)

# 求解神经网络参数

# 定义损失函数
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
# 定义训练过程
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 变量初始化
init = tf.global_variables_initializer()
# 定义Session
sess = tf.Session()
# 将网络结构图写到文件中
writer = tf.summary.FileWriter('logs/', sess.graph)
# 执行初始化工作
sess.run(init)

# 绘制求解的曲线
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

# 进行训练
for i in range(1000):
    # 执行训练，并传入数据
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 100 == 0:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass

        # print sess.run(loss, feed_dict = {xs: x_data, ys: y_data})
        # 计算预测值
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # 绘制预测值
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.1)
# 关闭Session
sess.close()
