import tensorflow as tf 
import numpy as np 

import matplotlib.pyplot as plt 
from pylab import mpl 
# 
mpl.rcParams["font.serif"] = ['simhei']



x_data = np.linspace(-1, 1, 250, dtype=np.float32)[:, np.newaxis] 

noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)

y_data = np.square(x_data) - 0.5*x_data + noise


xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

input_size_1 = 1
output_size_1 = 10

input_size_2 = 10
output_size_2 = 1


weights_1 = tf.Variable(tf.random_normal([input_size_1, output_size_1]))
weights_2 = tf.Variable(tf.random_normal([input_size_2, output_size_2]))


biases_1 = tf.Variable(tf.zeros([1, output_size_1]))
biases_2 = tf.Variable(tf.zeros([1, output_size_2]))


layer_1 = tf.nn.relu(tf.matmul(xs, weights_1) + biases_1)
prediction = tf.matmul(layer_1, weights_2) + biases_2

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)




def __loss():
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		x = sess.run(xs, feed_dict={xs: x_data})
		y = sess.run(ys, feed_dict={ys: y_data})
		a = 0
		x_layer = [1,2,3,4,5,6,7,8,9,10]

		for i in range(1200):
			sess.run(train_step, feed_dict={xs: x_data, ys: y_data})

			loss2 = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
			lay1 = sess.run(layer_1, feed_dict={xs: x_data})
			plt.scatter(i, loss2, marker='*')
		plt.ion()
		plt.title('Loss Results')
		plt.xlabel('Train Steps')
		plt.ylabel('Loss value')
		plt.show()
		plt.savefig('images/loss.png', format='png')
		plt.close()
def __result():
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		x = sess.run(xs, feed_dict={xs: x_data})
		y = sess.run(ys, feed_dict={ys: y_data})
		a = 0

		for i in range(1200):
			sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
			if i % 200 == 0:
				a += 1
				w1 = sess.run(weights_1)
				w2 = sess.run(weights_2)
				print("Weights_1 :{}".format(w1))
				print("weights_2 :{}".format(w2))
				# print(a)
				loss_1 = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
				print("Loss :{}".format(loss_1))
				pre = sess.run(prediction, feed_dict={xs: x_data})
				# plt.ion()
				# plt.figure(i)
				plt.subplot(2,3,a).set_title("Group{} results:".format(str(a)))
				plt.plot(x, pre, 'r')
				plt.scatter(x, y,s=2,c='b')
				plt.xlabel("x_data")
				plt.ylabel("y_data")
		plt.ion()
		plt.title("Results")
		plt.subplots_adjust(wspace=0.3, hspace=0.5)
		plt.show()
		plt.savefig('images/results.png', format='png')
		plt.close()

__loss()
__result()


	
		
			





