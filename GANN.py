import tensorflow as tf
import numpy as np

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def sigmoid(x):
    return 1/(1+tf.exp(-x))

def data():
    x = np.array([[20,50,2], [20,30,3], [20,20,4], [40,50,3], [40,30,2], [40,20,5], [40,50,4], [40,30,5], [40,20,2]])/np.array([100,100,5])
    y = np.array([15,4,1,18,21,1,17,5,18])/100
    return x, y

if __name__ == '__main__':

    x = tf.placeholder(tf.float32, shape=[None, 3])
    y = tf.placeholder(tf.float32, shape=[None, 1])

    W1 = weight_variable([3, 3])
    BH = bias_variable([3])
    W2 = weight_variable([3, 1])
    BO = bias_variable([1])
    
    y_ = sigmoid(tf.matmul(sigmoid(tf.matmul(x, W1) + BH), W2) + BO)

    loss = tf.reduce_mean(tf.square(y-y_))

    train = tf.train.GradientDescentOptimizer(1).minimize(loss)
    ix, iy = data()
    #print(ix[0].shape)
    #print(iy)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        predArr = None
        lossArr = 0
        for j in range(30000):
            sess.run(train, feed_dict={x: ix.reshape((9,3)), y: iy.reshape((9,1))})
            if j % 5000 == 0:
                predArr, lossArr = sess.run([y_, loss], feed_dict={x: ix.reshape((9,3)), y: iy.reshape((9,1))})
                print(lossArr)

        print(predArr*100)