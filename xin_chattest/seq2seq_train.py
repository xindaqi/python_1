import tensorflow as tf  # 0.12
import seq2seq_model
import os
import numpy as np
import math
 
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
 
train_encode_vec = './data/word2vec/train_question_encode.vec'
train_decode_vec = './data/word2vec/train_answer_decode.vec'
test_encode_vec = './data/word2vec/test_question_encode.vec'
test_decode_vec = './data/word2vec/test_answer_decode.vec'
 
# 词汇表大小5000
vocabulary_encode_size = 470
vocabulary_decode_size = 470
 
buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
layer_size = 256  # 每层大小
num_layers = 3   # 层数
batch_size = 64

def read_line():
	with open(train_encode_vec, 'r') as f:
		question = f.readline()
		print("question: {}".format(question))
		q_split = question.split()
		print("question split: {}".format(q_split))

# read_line()
 
# 读取*dencode.vec和*decode.vec数据（数据还不算太多, 一次读人到内存）
def read_data(question_path, answer_path, max_size=None):
	data_set = [[] for _ in buckets]


	with tf.gfile.GFile(question_path, mode="r") as question_file:
		with tf.gfile.GFile(answer_path, mode="r") as answer_file:
			question, answer = question_file.readline(), answer_file.readline()
			counter = 0
			while question and answer and (not max_size or counter < max_size):
				counter += 1
				question_ids = [int(x) for x in question.split()]
				answer_ids = [int(x) for x in answer.split()]
				answer_ids.append(EOS_ID)
				for bucket_id, (question_size, answer_size) in enumerate(buckets):
					if len(question_ids) < question_size and len(answer_ids) < answer_size:
						data_set[bucket_id].append([question_ids, answer_ids])
						break
				question, answer = question_file.readline(), answer_file.readline()
				print("question: {}, answer: {}".format(question, answer))
	return data_set

# train_set = read_data(train_encode_vec, train_decode_vec)
# print("Train set: {}".format(train_set))
# train_bucket_sizes = [len(train_set[b]) for b in range(len(buckets))]
# print("train bucket sizes: {}".format(train_bucket_sizes))
# total_size = float(sum(train_bucket_sizes))
# print("total size: {}".format(total_size))
# for i in train_set:
# 	print("data: {}".format(i))
# 	print("train data sizes: {}".format(len(i)))
# print("lenght of data set: {}".format(len(train_set)))

# train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / total_size for i in range(len(train_bucket_sizes))]
# print("train buckets scale: {}".format(train_buckets_scale))

model = seq2seq_model.Seq2SeqModel(question_vocab_size=vocabulary_encode_size, answer_vocab_size=vocabulary_decode_size,
                                   buckets=buckets, size=layer_size, num_layers=num_layers, max_gradient_norm= 5.0,
                                   batch_size=batch_size, learning_rate=0.5, learning_rate_decay_factor=0.97, forward_only=False)



# # config = tf.ConfigProto()
# # config.gpu_options.allocator_type = 'BFC'  # 防止 out of memory
if __name__ == "__main__":
	with tf.Session() as sess:
	 
	# with tf.Session(config=config) as sess:
		# 恢复前一次训练
		# ckpt = 
		ckpt = tf.train.get_checkpoint_state('./models')
		if ckpt != None:
			train_turn = ckpt.model_checkpoint_path.split('-')[1]
			print("model path: {}, train turns: {}".format(ckpt.model_checkpoint_path, train_turn))
			model.saver.restore(sess, ckpt.model_checkpoint_path)
		else:
			sess.run(tf.global_variables_initializer())
	 
		train_set = read_data(train_encode_vec, train_decode_vec)
		test_set = read_data(test_encode_vec, test_decode_vec)
	 
		train_bucket_sizes = [len(train_set[b]) for b in range(len(buckets))]
		# print("train bucket sizes: {}".format(train_bucket_sizes))
		train_total_size = float(sum(train_bucket_sizes))
		train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in range(len(train_bucket_sizes))]
	 
		loss = 0.0
		total_step = int(train_turn)
		previous_losses = []
		# 一直训练，每过一段时间保存一次模型
		while True:
			random_number_01 = np.random.random_sample()
			# get minimum i as bucket id when value > randmom value
			bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])
	 
			encoder_inputs, decoder_inputs, answer_weights = model.get_batch(train_set, bucket_id)
			# print("encoder inputs: {}, decoder inputs: {}, answer weights: {}".format(encoder_inputs, decoder_inputs, answer_weights))
			_, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, answer_weights, bucket_id, False)
	 
			loss += step_loss / 500
			total_step += 1
	 
			print("total step: {}".format(total_step))
			if total_step % 500 == 0:
				print("global step: {}, learning rate: {}, loss: {}".format(model.global_step.eval(), model.learning_rate.eval(), loss))
	 
				# 如果模型没有得到提升，减小learning rate
				if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
					sess.run(model.learning_rate_decay_op)
				previous_losses.append(loss)
				# 保存模型
				checkpoint_path = "./models/chatbot_seq2seq.ckpt"
				model.saver.save(sess, checkpoint_path, global_step=model.global_step)
				loss = 0.0
				# 使用测试数据评估模型
				for bucket_id in range(len(buckets)):
					if len(test_set[bucket_id]) == 0:
						continue
					encoder_inputs, decoder_inputs, answer_weights = model.get_batch(test_set, bucket_id)
					_, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, answer_weights, bucket_id, True)
					eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
					print("bucket id: {}, eval ppx: {}".format(bucket_id, eval_ppx))