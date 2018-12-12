import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

x_data = np.linspace(-1, 1, 250, dtype=np.float32)[:, np.newaxis] 

noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)

y_data = np.square(x_data) - 0.5*x_data + noise

print("Type of x_data : {}".format(type(x_data)))
print("Shape of x_data: {}".format(x_data.shape))

def loadModel():
	with tf.Session() as sess:
		# 载入模型,后去模型的图graph
		saver = tf.train.import_meta_graph('models/model.ckpt-299.meta')
		# 载入模型变量
		saver.restore(sess, tf.train.latest_checkpoint('models/'))
		# 获取新增变量
		pre = tf.get_collection('prediction')[0]
		# 获取输入变量
		graph = tf.get_default_graph()
		x = graph.get_operation_by_name('x').outputs[0]
		y = graph.get_operation_by_name('y').outputs[0]
		predictionModel = sess.run(pre, feed_dict={x: x_data, y: y_data})
		print("Shape of prediction value from model: {}".format(predictionModel.shape))
		plt.ion()
		# plt.figure(2)
		plt.title("Load Model for Transfer Training")

		plt.scatter(x_data, y_data, s=2, c='b', label='Real')
		plt.plot(x_data, predictionModel, 'r', label='Predict')
		plt.legend(loc='upper right')
		plt.xlabel("x/cm")
		plt.ylabel('Prediction&Reality/cm')

		plt.show()
		plt.savefig('images/loadModelPredict.png', format='png')
		# 获取模型变量
		print("Load Weights_1 from model: {}".format(sess.run('weights_1:0')))
		print("Load Weights_2 from model: {}".format(sess.run('weights_2:0')))
		print("Load Biases_1 from model: {}".format(sess.run('biases_1:0')))
		print("Load Biases_2 from model: {}".format(sess.run('biases_2:0')))
loadModel()
