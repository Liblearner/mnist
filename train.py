import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import math
import time
import matplotlib.pyplot as plt
from tensorflow.python.framework import graph_util
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
train_dir = './model'

# MNIST数据集相关参数
num_classes = 10  # 总类别数（0-9）
img_rows, img_cols = 28, 28  # 图像尺寸

# load dataset
def load_dataset():
    # 划分训练集、测试集
    data = h5py.File("dataset//data_notwhite.h5", "r")
    X_data = np.array(data['X'])  # data['X']是h5py._hl.dataset.Dataset类型，转化为array
    Y_data = np.array(data['Y'])
    num, _, _ = X_data.shape
    X_data = X_data.reshape(num, 40, 40, 1)
    print(type(X_data))
    X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, train_size=0.9, test_size=0.1, random_state=22)
    # print(X_train.shape)
    # print(y_train[456])
    # image = Image.fromarray(X_train[456])
    # image.show()
    # y_train = y_train.reshape(1,y_train.shape[0])
    # y_test = y_test.reshape(1,y_test.shape[0])
    print(X_train.shape)
    # print(X_train[0])
    X_train = X_train / 255.  # 归一化
    X_test = X_test / 255.
    # print(X_train[0])
    # one-hot
    y_train = np_utils.to_categorical(y_train, num_classes=14)
    print(y_train.shape)
    y_test = np_utils.to_categorical(y_test, num_classes=14)
    print(y_test.shape)

    return X_train, X_test, y_train, y_test

# 加载mnist数据集并进行预处理
def load_mnist():
    # 加载数据集
    # 数据集位置：C:\Users\Lenovo\.keras\datasets\mnist.npz
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    # 归一化数据
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # 转换标签为one-hot编码
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test

def weight_variable(shape):
    tf.compat.v1.set_random_seed(1)
    return tf.Variable(tf.random.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.0, shape=shape))


def conv2d(x, W):
    # return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(z):
    # return tf.nn.max_pool2d(z, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # verilog实现时是没有填充的
    return tf.nn.max_pool2d(z, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

def avg_pool_4x4(z):
    return tf.nn.avg_pool2d(z, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')


def random_mini_batches(X, Y, mini_batch_size=100, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))
    print("shuffled done")

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def learning_curve(train_acc, test_acc, stride=20):
    x = len(train_acc)
    xlen = x * stride
    xLim = np.arange(0, xlen, stride)
    plt.figure()
    plt.plot(xLim, train_acc, color='r', label='Training acc')
    plt.plot(xLim, test_acc, color='b', label='Testing acc')
    plt.legend()
    plt.show()


def cnn_model(X_train, y_train, X_test, y_test, keep_prob, lamda, num_epochs=450, minibatch_size=100, learning_rate = 0.001):
    X = tf.compat.v1.placeholder(tf.float32, [None, 40, 40, 1], name="input_x")
    y = tf.compat.v1.placeholder(tf.float32, [None, 14], name="input_y")
    kp = tf.compat.v1.placeholder_with_default(1.0, shape=(), name="keep_prob")
    lam = tf.compat.v1.placeholder(tf.float32, name="lamda")

    # conv1
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    z1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)
    maxpool1 = max_pool_2x2(z1)  # max_pool1完后maxpool1维度为[?,20,20,32]

    # conv2
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    z2 = tf.nn.relu(conv2d(maxpool1, W_conv2) + b_conv2)
    maxpool2 = max_pool_2x2(z2)  # max_pool2,shape [?,10,10,64]

    # conv3  效果比较好的一次模型是没有这一层，只有两次卷积层，隐藏单元100，训练20次
    # W_conv3 = weight_variable([5, 5, 64, 128])
    # b_conv3 = bias_variable([128])
    # z3 = tf.nn.relu(conv2d(maxpool2, W_conv3) + b_conv3)
    # maxpool3 = max_pool_2x2(z3)  # max_pool3,shape [?,8,8,128]

    # full connection1
    W_fc1 = weight_variable([10 * 10 * 64, 512])
    b_fc1 = bias_variable([512])
    maxpool2_flat = tf.reshape(maxpool2, [-1, 10 * 10 * 64])
    z_fc1 = tf.nn.relu(tf.matmul(maxpool2_flat, W_fc1) + b_fc1)
    # z_fc1_drop = tf.nn.dropout(z_fc1, keep_prob=kp)
    z_fc1_drop = tf.nn.dropout(z_fc1, rate = 1-kp)
    # softmax layer
    W_fc2 = weight_variable([512, 14])
    b_fc2 = bias_variable([14])
    z_fc2 = tf.add(tf.matmul(z_fc1_drop, W_fc2), b_fc2, name="outlayer")
    prob = tf.nn.softmax(z_fc2, name="probability")

    # cost function
    regularizer = tf.contrib.layers.l2_regularizer(lam)  # l2正则化,防止过拟合
    regularization = regularizer(W_fc1) + regularizer(W_fc2)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z_fc2)) + regularization

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(cost)
    # train = tf.compat.v1.train.AdamOptimizer().minimize(cost)
    # output_type='int32', name="predict"
    pred = tf.argmax(prob, 1, output_type="int32", name="predict")  # 输出结点名称predict方便后面保存为pb文件
    correct_prediction = tf.equal(pred, tf.argmax(y, 1, output_type='int32'))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.set_random_seed(1)  # to keep consistent results

    seed = 0
    acc = 0.97
    train_accs = []
    test_accs = []
    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        # sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))
        sess.run(init)
        step = 0

        for epoch in range(num_epochs):
            seed = seed + 1
            epoch_cost = 0.
            num_minibatches = int(X_train.shape[0] / minibatch_size)
            minibatches = random_mini_batches(X_train, y_train, minibatch_size, seed)
            minibatchesTest = random_mini_batches(X_test, y_test, minibatch_size, seed)
            test_i = 0
            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch

                _, minibatch_cost = sess.run([train, cost],
                                             feed_dict={X: minibatch_X, y: minibatch_Y, kp: keep_prob, lam: lamda})
                epoch_cost += minibatch_cost / num_minibatches
                step = step + 1
                if (step % 20 == 0):
                    (minibatchtest_X, minibatchtest_Y) = minibatchesTest[test_i]
                    test_i = test_i + 1
                    test_acc = accuracy.eval(feed_dict={X: minibatchtest_X, y: minibatchtest_Y, lam: lamda})
                    train_acc = accuracy.eval(feed_dict={X: minibatch_X, y: minibatch_Y, lam: lamda})
                    train_accs.append(train_acc)
                    test_accs.append(test_acc)
                    print("test accuracy", test_acc)
                    print("cost", minibatch_cost)
                    if test_acc > acc:
                        acc = test_acc
                        # saver = tf.train.Saver(
                        #     {'W_conv1': W_conv1, 'b_conv1': b_conv1, 'W_conv2': W_conv2, 'b_conv2': b_conv2,
                        #      'W_fc1': W_fc1, 'b_fc1': b_fc1, 'W_fc2': W_fc2, 'b_fc2': b_fc2})
                        saver = tf.compat.v1.train.Saver(
                            {'W_conv1': W_conv1, 'b_conv1': b_conv1, 'W_conv2': W_conv2, 'b_conv2': b_conv2,
                             'W_fc1': W_fc1, 'b_fc1': b_fc1, 'W_fc2': W_fc2, 'b_fc2': b_fc2})
                        checkpoint_path = os.path.join(train_dir, 'thing.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)
            print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            print(str((time.strftime('%Y-%m-%d %H:%M:%S'))))

        # 这个accuracy是前面的accuracy，tensor.eval()和Session.run区别很小
        train_acc = accuracy.eval(feed_dict={X: X_train[:1000], y: y_train[:1000], kp: 0.8, lam: lamda})
        print("train accuracy", train_acc)
        test_acc = accuracy.eval(feed_dict={X: X_test[:1000], y: y_test[:1000], lam: lamda})
        print("test accuracy", test_acc)

        # save model
        saver = tf.compat.v1.train.Saver({'W_conv1': W_conv1, 'b_conv1': b_conv1, 'W_conv2': W_conv2, 'b_conv2': b_conv2,
                                'W_fc1': W_fc1, 'b_fc1': b_fc1, 'W_fc2': W_fc2, 'b_fc2': b_fc2})
        saver.save(sess, "model//cnn_model.ckpt")

        # 将训练好的模型保存为.pb文件，方便在Android studio中使用
        output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess,
                                                                                  sess.graph_def,
                                                                     output_node_names=['predict'])
        with tf.gfile.FastGFile('model//digital_gesture.pb', mode='wb') as f:  # ’wb’中w代表写文件，b代表将数据以二进制方式写入文件。
            f.write(output_graph_def.SerializeToString())
        learning_curve(train_accs, test_accs, 20)

def lenet_model(X_train, y_train, X_test, y_test, keep_prob, lamda, num_epochs, minibatch_size,learning_rate=0.001):
    # 定义anPH1A例程中的小网络结构
    X = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1], name="input_x")
    y = tf.compat.v1.placeholder(tf.float32, [None, 10], name="input_y")
    kp = tf.compat.v1.placeholder_with_default(1.0, shape=(), name="keep_prob")
    lam = tf.compat.v1.placeholder(tf.float32, name="lamda")

    #pre_relu，采用平均池化
    # z0 = avg_pool_4x4(X)
    # conv1
    W_conv1 = weight_variable([5, 5, 1, 3])
    b_conv1 = bias_variable([3])
    z1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)
    maxpool1 = max_pool_2x2(z1)  # max_pool1完后maxpool1维度为[?,12,12,3]

    # conv2
    W_conv2 = weight_variable([5, 5, 3, 3])
    b_conv2 = bias_variable([3])
    z2 = tf.nn.relu(conv2d(maxpool1, W_conv2) + b_conv2)
    maxpool2 = max_pool_2x2(z2)  # max_pool2,shape [?,4,4,3]

    # full connection1
    W_fc1 = weight_variable([4 * 4 * 3, 10])
    b_fc1 = bias_variable([10])
    maxpool2_flat = tf.reshape(maxpool2, [-1, 4 * 4 * 3])
    z_fc1 = tf.nn.relu(tf.matmul(maxpool2_flat, W_fc1) + b_fc1)
    prob = tf.nn.softmax(z_fc1, name="probability")

    # cost function
    regularizer = tf.contrib.layers.l2_regularizer(lam)  # l2正则化,防止过拟合
    regularization = regularizer(W_fc1)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z_fc1)) + regularization

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(cost)
    # train = tf.compat.v1.train.AdamOptimizer().minimize(cost)
    # output_type='int32', name="predict"
    pred = tf.argmax(prob, 1, output_type="int32", name="predict")  # 输出结点名称predict方便后面保存为pb文件
    correct_prediction = tf.equal(pred, tf.argmax(y, 1, output_type='int32'))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.set_random_seed(1)  # to keep consistent results

    seed = 0
    acc = 0.97
    train_accs = []
    test_accs = []
    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        # sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))
        sess.run(init)
        step = 0

        for epoch in range(num_epochs):
            seed = seed + 1
            epoch_cost = 0.
            num_minibatches = int(X_train.shape[0] / minibatch_size)
            minibatches = random_mini_batches(X_train, y_train, minibatch_size, seed)
            minibatchesTest = random_mini_batches(X_test, y_test, minibatch_size, seed)
            test_i = 0
            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch

                _, minibatch_cost = sess.run([train, cost],
                                             feed_dict={X: minibatch_X, y: minibatch_Y, kp: keep_prob, lam: lamda})
                epoch_cost += minibatch_cost / num_minibatches
                step = step + 1
                if (step % 20 == 0):
                    (minibatchtest_X, minibatchtest_Y) = minibatchesTest[test_i]
                    test_i = test_i + 1
                    test_acc = accuracy.eval(feed_dict={X: minibatchtest_X, y: minibatchtest_Y, lam: lamda})
                    train_acc = accuracy.eval(feed_dict={X: minibatch_X, y: minibatch_Y, lam: lamda})
                    train_accs.append(train_acc)
                    test_accs.append(test_acc)
                    print("test accuracy", test_acc)
                    print("cost", minibatch_cost)
                    if test_acc > acc:
                        acc = test_acc
                        # saver = tf.train.Saver(
                        #     {'W_conv1': W_conv1, 'b_conv1': b_conv1, 'W_conv2': W_conv2, 'b_conv2': b_conv2,
                        #      'W_fc1': W_fc1, 'b_fc1': b_fc1, 'W_fc2': W_fc2, 'b_fc2': b_fc2})
                        saver = tf.compat.v1.train.Saver(
                            {'W_conv1': W_conv1, 'b_conv1': b_conv1, 'W_conv2': W_conv2, 'b_conv2': b_conv2,
                             'W_fc1': W_fc1, 'b_fc1': b_fc1})
                        checkpoint_path = os.path.join(train_dir, 'thing.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)
            print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            print(str((time.strftime('%Y-%m-%d %H:%M:%S'))))

        # 这个accuracy是前面的accuracy，tensor.eval()和Session.run区别很小
        train_acc = accuracy.eval(feed_dict={X: X_train[:1000], y: y_train[:1000], kp: 0.8, lam: lamda})
        print("train accuracy", train_acc)
        test_acc = accuracy.eval(feed_dict={X: X_test[:1000], y: y_test[:1000], lam: lamda})
        print("test accuracy", test_acc)

        # save model
        saver = tf.compat.v1.train.Saver({'W_conv1': W_conv1, 'b_conv1': b_conv1, 'W_conv2': W_conv2, 'b_conv2': b_conv2,
                                'W_fc1': W_fc1, 'b_fc1': b_fc1})
        saver.save(sess, "model//lenet_model.ckpt")

        # 将训练好的模型保存为.pb文件，方便在Android studio中使用
        output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess,
                                                                                  sess.graph_def,
                                                                     output_node_names=['predict'])
        with tf.gfile.FastGFile('model//lenet_model.pb', mode='wb') as f:  # ’wb’中w代表写文件，b代表将数据以二进制方式写入文件。
            f.write(output_graph_def.SerializeToString())
        learning_curve(train_accs, test_accs, 20)


if __name__ == "__main__":
    print("载入数据集: " + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
    X_train, X_test, y_train, y_test = load_mnist()
    print("开始训练: " + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
    lenet_model(X_train, y_train, X_test, y_test, 0.7, 0.000001, num_epochs=100, minibatch_size=200, learning_rate = 0.0001)
    print("训练结束: " + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
