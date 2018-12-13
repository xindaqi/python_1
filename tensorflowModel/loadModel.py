import tensorflow as tf
import os

def saveModel():
	MODEL_SAVE_PATH = './models'
	MODEL_NAME = 'model.ckpt'
	v1 = tf.Variable(tf.constant(2.0, shape=[1]), name='v1')
	v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
	result = v1 + v2
	# 添加模型图变量
	tf.add_to_collection('result', result)
	saver = tf.train.Saver()
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))



def loadModel():
	with tf.Session() as sess:
		# 加载模型的图
		saver = tf.train.import_meta_graph("models/model.ckpt.meta")
		# 加载保存的模型
		saver.restore(sess, tf.train.latest_checkpoint('models/'))
		
		# 获取模型变量
		res = tf.get_collection('result')[0]
		print("Load model to get variable v1: {}".format(sess.run('v1:0')))
		print("Load model to get variable v2: {}".format(sess.run('v2:0')))
		print("Load model to get varaible result: {}".format(sess.run(res)))

saveModel()
loadModel()