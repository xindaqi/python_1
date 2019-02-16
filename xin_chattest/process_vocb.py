import os
import random
import re
 
conv_path = './data/dgk_shooter_min.conv'

print(os.path.exists(conv_path))
 
if not os.path.exists(conv_path):
	print('数据集不存在')
	exit()
 
# chinese trans to utf8 format
convs = []  # 对话集合

with open(conv_path, encoding = "utf8") as f:
	print("source data: {}".format(f))
	one_conv = []        # 一次完整对话
	for line in f:
		# line = line.strip('\n').replace('/', '')
		line = line.strip('\n')
		line = re.sub("[\s+\.\!\/_,$%?^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",line)
		

		# print("Line vocabulary: {}".format(line))
		if line == '':
			continue
		if line[0] == 'E':
			if one_conv:
				convs.append(one_conv)
			# initialize one_conv and clear
			one_conv = []
		elif line[0] == 'M':
			# cut string and trans to list
			# get element 1 namely conversation content expect M
			# one_conv.append(line.split(' ')[1])
			one_conv.append(line[1:])
			print("One conversation: {}".format(one_conv))
			print("Full conversations: {}".format(convs))

	print("Full conversations: {}".format(convs))
	print("Length of full conversations: {}".format(len(convs)))
	# for conv in convs_temp:
	# 	print("Type of conv: {}".format(type(conv)))

		# conv = re.sub("[\s+\.\!\/_,$%?^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",conv)
		# convs.append(conv)

	# print("Full delete symbol: {}".format(convs))

"""
print(convs[:3])  # 个人感觉对白数据集有点不给力啊
[ ['畹华吾侄', '你接到这封信的时候', '不知道大伯还在不在人世了'], 
  ['咱们梅家从你爷爷起', '就一直小心翼翼地唱戏', '侍奉宫廷侍奉百姓', '从来不曾遭此大祸', '太后的万寿节谁敢不穿红', '就你胆儿大', '唉这我舅母出殡', '我不敢穿红啊', '唉呦唉呦爷', '您打得好我该打', '就因为没穿红让人赏咱一纸枷锁', '爷您别给我戴这纸枷锁呀'], 
  ['您多打我几下不就得了吗', '走', '这是哪一出啊 ', '撕破一点就弄死你', '唉', '记着唱戏的再红', '还是让人瞧不起', '大伯不想让你挨了打', '还得跟人家说打得好', '大伯不想让你再戴上那纸枷锁', '畹华开开门哪'], ....]
"""




 
# 把对话分成问与答
questions = [] # 问
answers = []   # 答
for conv in convs:
	if len(conv) == 1:
		continue
	if len(conv) % 2 != 0:  # 奇数对话数, 转为偶数对话
		conv = conv[:-1]
	for i in range(len(conv)):
		if i % 2 == 0:
			questions.append(conv[i])
		else:
			answers.append(conv[i])

print("Questions: {}".format(questions))

print("Answers: {}".format(answers))
print("Lenght of questions: {}".format(len(questions)))
print("Lenght of answers: {}".format(len(answers)))
 
"""
print(len(ask), len(response))
print(ask[:3])
print(response[:3])
['畹华吾侄', '咱们梅家从你爷爷起', '侍奉宫廷侍奉百姓']
['你接到这封信的时候', '就一直小心翼翼地唱戏', '从来不曾遭此大祸']
"""
 
def question_answer_dataset(questions, answers, TESTSET_SIZE = 50):
    # 创建文件
    question_train_enc = open('./data/question_train.enc','w')  # 问
    answer_train_dec = open('./data/answer_train.dec','w')  # 答
    question_test_enc  = open('./data/question_test.enc', 'w')  # 问
    answer_test_dec  = open('./data/answer_test.dec', 'w')  # 答
 

    # get test question conversations number in casual
    test_index = random.sample([i for i in range(len(questions))],TESTSET_SIZE)
 
    for i in range(len(questions)):
        if i in test_index:
            question_test_enc.write(questions[i]+'\n')
            answer_test_dec.write(answers[i]+ '\n' )
        else:
            question_train_enc.write(questions[i]+'\n')
            answer_train_dec.write(answers[i]+ '\n' )
        if i % 1000 == 0:
            print(len(range(len(questions))), '处理进度:', i)
 
    question_train_enc.close()
    answer_train_dec.close()
    question_test_enc.close()
    answer_test_dec.close()
 
question_answer_dataset(questions, answers)
