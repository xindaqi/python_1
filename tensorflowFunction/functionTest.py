import tensorflow as tf 

def varScope():
	with tf.Session() as sess:
		# 定义变量空间
		with tf.variable_scope("Input_1"):
			# 定义常量变量
			v1 = tf.constant([1.0], shape=[1], name='v1')
			print("V1 name is: {}".format(v1.name))
			print("V1 value is: {}".format(sess.run(v1)))
		# 定义命名空间
		with tf.name_scope("Input_2"):
			# 定义常量变量
			v1 = tf.constant([1.0], shape=[1], name='v1')
			print("v1 name is: {}".format(v1.name))
			print("v1 value is: {}".format(sess.run(v1)))

def varDefine():
	# 定义变量空间
	with tf.variable_scope("Input_3"):
		# 定义变量,指定变量名v1,初始化维度[1]和变量值initializer
		v1 = tf.get_variable("v1", [1], initializer=tf.constant_initializer(1))	
		# 定义变量,初始化变量值[1],初始化维度[1],指定变量名v2
		v2 = tf.Variable([1], shape=[1],name='v2')
			
	with tf.Session() as sess:
		# 初始化所有变量
		init_op = tf.global_variables_initializer()
		# 运行初始化
		sess.run(init_op)
		print("v1 name is: {}".format(v1.name))
		print("v1 value is: {}".format(sess.run(v1)))
		print("v2 name is: {}".format(v2.name))
		print("v2 value is: {}".format(sess.run(v2)))

def regularizerFun():
	w = tf.constant([[1.0, -1.0], [-2.0, 2.0]])
	with tf.Session() as sess:
		# L1正则化,L1=lambda*∑|w|
		L1 = tf.contrib.layers.l1_regularizer(0.2)(w)
		# L2正则化,L2=lambda*[(∑|w^2|)/2]
		L2 = tf.contrib.layers.l2_regularizer(0.2)(w)

		print("L1 regularizer result: {}".format(sess.run(L1)))
		print("L2 regularizer result: {}".format(sess.run(L2)))


def exponeitialMovingAverage():
	v1 = tf.Variable(0, dtype=tf.float32, name='v')
	step = tf.Variable(0, trainable=False)
	# 初始衰减率0.99, 控制衰减率变量step
	ema = tf.train.ExponentialMovingAverage(0.99, step)
	# 更新moving average
	maintain_averages_op = ema.apply([v1])
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		# v1=0, 初始化shadow=0
		v1_1, shadow_1 = sess.run([v1, ema.average(v1)])
		print("v1 value is: {}, shadow variable value is: {}".format(v1_1, shadow_1))
		# v1=5
		sess.run(tf.assign(v1, 5))
		# 更新shadow
		sess.run(maintain_averages_op)
		# v1_2=5, shadow_2=decay*shadow + (1-decay)*variable
		# shadow=0,step-=0,decay=min{0.99,(1+0)/(10+0)}=0.1
		# shadow_2=0.1*0 + (1-0.1)*5=4.5
		v1_2, shadow_2 = sess.run([v1, ema.average(v1)])
		print("v1 value is: {}, shadow variable is: {}".format(v1_2, shadow_2))
		# 更新numUpdates=100
		sess.run(tf.assign(step, 100))
		# 更新v1=8
		sess.run(tf.assign(v1, 8))
		# 更新shadow=4.5
		sess.run(maintain_averages_op)
		# v1_3=8, step=100, decay=min{0.99,(1+100)/(10+100)}=0.9182
		# shadow_3=decay*shadow_2+(1-decay)*variable
		# shadow_3=0.9182*4.5 + (1-0.9182)*8=4.7863
		v1_3, step, shadow_3 = sess.run([v1, step, ema.average(v1)])
		print("v1 value is: {}, step value is: {}, shadow variable value is: {}".format(v1_3, step, shadow_3))



varScope()
varDefine()
regularizerFun()
exponeitialMovingAverage()


