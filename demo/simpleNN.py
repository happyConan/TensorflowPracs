#coding: utf-8
"""
-------------------------------------------------
   File Name：     simpleNN
   Description :
   Author :       WANGFEI
   date：          2018/8/18
-------------------------------------------------
"""
import tensorflow as tf
import numpy as np
BATCH_SIZE=8
seed=23455
rng=np.random.RandomState(seed)

with tf.name_scope('data'):
    X = rng.rand(32, 2)
    Y = [[int(x1 + x2 < 1)] for [x1, x2] in X]
    print('X:\n', X)
    print('Y:\n', Y)

x=tf.placeholder(tf.float32,shape=[None,2])
y_=tf.placeholder(tf.float32,shape=[None,1])

with tf.name_scope('parameter'):
    w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
    w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
    tf.summary.histogram('w1', w1)
    tf.summary.histogram('w2', w2)

with tf.name_scope('prediction'):
    a=tf.matmul(x,w1)
    y=tf.matmul(a,w2)

with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.square(y-y_))
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    with tf.name_scope('init'):
        init_op=tf.global_variables_initializer()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(init_op)
    for i in range(200):
        start=(i*BATCH_SIZE)%32
        end=start+BATCH_SIZE
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        rs = sess.run(merged,feed_dict={x:X[start:end],y_:Y[start:end]})
        writer.add_summary(rs, i)
        if i%200==0:
            loss_val=sess.run(loss, feed_dict={x: X, y_: Y})
            print('%d轮后，损失为：%g'%(i,loss_val))


