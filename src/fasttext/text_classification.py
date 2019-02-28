# coding:utf-8
import fasttext as ft

# 官方文档：https://fasttext.cc/     https://pypi.org/project/fasttext/
# git https://github.com/facebookresearch/fastText/
# Demo https://github.com/pyk/fastText.py

"""
分类算法原理：
    与cbow模型根据上下文预测中间词的算法类似，只不过在分类任务中是根据整个序列预测分类
    在cbow模型基础上进行了优化：
        引入n-gram相关理论，用n-gram的向量代替简单的词向量,一定程度上解决了词序信息丢失的问题
        对于简单的词向量来说：“我爱他” 和 “他爱我”的表示是一样的；引入2-gram之后，则前者会表示为
        我、爱、他、我爱、爱他；后者会表示为  他、爱、我、他爱、爱我  这五个向量求和，就能比较明确
        的表示词序信息
"""
input_file = '../../data/fastText/train.txt'
output = '../../model/fastText/classify.model'
"""
优化方式：
    优化数据，如统一成小写字符，去掉停用词、高频词、低频词、特殊字符等
    训练更多的轮次
    增加学习率
    用n-gram 代替unigram
    增加词向量维度
"""
# set params
dim = 10  # 词向量的维度
lr = 0.05  # 学习率
epoch = 500  # 迭代次数
min_count = 1  # 最小词频数
word_ngrams = 6  # 词语的前后关联程度
bucket = 2000000
thread = 4
silent = 0  # 是否打印进度
label_prefix = '__label__'
classifier = ft.supervised(input_file, output, dim=dim, lr=lr, epoch=epoch,
                           min_count=min_count, word_ngrams=word_ngrams, bucket=bucket,
                           thread=thread, silent=silent, label_prefix=label_prefix)

result = classifier.test(input_file)
print('P@1:', result.precision)
print('R@1:', result.recall)
print('Number of examples:', result.nexamples)

# 预测
model = ft.load_model(output + '.bin')
texts = [
    'I am dumbfounded that I actually sat and watched this. I love independent films, horror films, and the whole zombie thing in general. But when you add ninga\'s, you\'ve crossed a line that should never be crossed. I hope the people in this movie had a great time making it, then at least it wasn\'t a total waste. You\'d never know by watching it though. Script? Are you kidding. Acting? I think even the trees were faking. Cinematography? Well, there must\'ve been a camera there. Period. I don\'t think there was any actual planning involved in the making of this movie. Such a total waste of time that I won\'t prolong it by commenting further.',
    'I loved this movie! It was all I could do not to break down into tears while watching it, but it is really very uplifting. I was struck by the performance of Ray Liotta, but especially the talent of Tom Hulce portraying Ray\'s twin brother who is mentally slow due to a tragic and terrible childhood event. But Tom\'s character, though heartbreaking, knows no self pity and is so full of hope and life. This is a great movie, don\'t miss it!!']
labels = model.predict(texts)
print(labels)
