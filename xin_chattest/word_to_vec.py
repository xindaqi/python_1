# 前一步生成的问答文件路径
# train_encode_file = 'train.enc'
# train_decode_file = 'train.dec'
# test_encode_file = 'test.enc'
# test_decode_file = 'test.dec'


question_train_encode_file = './data/question_train.enc'  # 问
answer_train_decode_file = './data/answer_train.dec' # 答
question_test_encode_file = './data/question_test.enc' # 问
answer_test_decode_file  = './data/answer_test.dec'  # 答
 
print('开始创建词汇表...')
# 特殊标记，用来填充标记对话
PAD = "__PAD__"
GO = "__GO__"
EOS = "__EOS__"  # 对话结束
UNK = "__UNK__"  # 标记未出现在词汇表中的字符
START_VOCABULART = [PAD, GO, EOS, UNK]
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
# 参看tensorflow.models.rnn.translate.data_utils
 
vocabulary_size = 5000
# 生成词汇表文件
def gen_vocabulary_file(input_file, output_file):
	vocabulary = {}
	with open(input_file) as f:
		counter = 0
		for line in f:
			counter += 1
			tokens = [word for word in line.strip()]
			for word in tokens:
				if word in vocabulary:
					vocabulary[word] += 1
				else:
					vocabulary[word] = 1

		# print("vocabulary: {}".format(vocabulary))
		vocabulary_list = START_VOCABULART + sorted(vocabulary, key=vocabulary.get, reverse=True)

		# print("vocabulary list: {}".format(vocabulary_list))
		# 取前5000个常用汉字, 应该差不多够用了(额, 好多无用字符, 最好整理一下. 我就不整理了)
		# if len(vocabulary_list) > 5000:
		# 	vocabulary_list = vocabulary_list[:5000]
		print(input_file + " 词汇表大小:", len(vocabulary_list))
		with open(output_file, "w") as ff:
			for word in vocabulary_list:
				ff.write(word + "\n")
 
gen_vocabulary_file(question_train_encode_file, "./data/word2vec/train_question_encode_vocabulary")
gen_vocabulary_file(answer_train_decode_file, "./data/word2vec/train_answer_decode_vocabulary")
 
train_question_encode_vocabulary_file = './data/word2vec/train_question_encode_vocabulary'
train_answer_decode_vocabulary_file = './data/word2vec/train_answer_decode_vocabulary'
 
print("对话转向量...")
# 把对话字符串转为向量形式
def convert_to_vector(input_file, vocabulary_file, output_file):
	tmp_vocab = []
	with open(vocabulary_file, "r") as f:
		# print("vocabulary: {}".format(f.readlines()))
		tmp_vocab.extend(f.readlines())
		print("temporay vocabulary: {}".format(tmp_vocab))
	tmp_vocab = [line.strip() for line in tmp_vocab]
	print("strip temporay vocabulary: {}".format(tmp_vocab))
	vocab = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])
	print("Dictionary vocabulary: {}".format(vocab))
	#{'硕': 3142, 'v': 577, 'Ｉ': 4789, '\ue796': 4515, '拖': 1333, '疤': 2201 ...}
	output_f = open(output_file, 'w')
	with open(input_file, 'r') as f:
		for line in f:
			line_vec = []
			print("Line: {}".format(line))
			for words in line.strip():
				# print("words: {}".format(words))
				line_vec.append(vocab.get(words, UNK_ID))
			print("Line vector: {}".format(line_vec))
			output_f.write(" ".join([str(num) for num in line_vec]) + "\n")
	output_f.close()
 
convert_to_vector(question_train_encode_file, train_question_encode_vocabulary_file, './data/word2vec/train_question_encode.vec')
convert_to_vector(answer_train_decode_file, train_answer_decode_vocabulary_file, './data/word2vec/train_answer_decode.vec')
 
convert_to_vector(question_test_encode_file, train_question_encode_vocabulary_file, './data/word2vec/test_question_encode.vec')
convert_to_vector(answer_test_decode_file, train_answer_decode_vocabulary_file, './data/word2vec/test_answer_decode.vec')