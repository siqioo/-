from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preData import preprocess_data

data_file = 'data.txt'  # 数据文件路径
x_train, x_test, y_train, y_test, spam_data, ham_data= preprocess_data(data_file)

#词袋模型
count_vector = CountVectorizer(stop_words='english')
#学习词汇词典并返回术语 - 文档矩阵(稀疏矩阵)。
train_data = count_vector.fit_transform(x_train)
# 使用符合fit的词汇表或提供给构造函数的词汇表，从原始文本文档中提取词频，转换成词频矩阵
test_data = count_vector.transform(x_test)

# 创建朴素贝叶斯分类器
classifier = MultinomialNB()

# 在训练数据上训练分类器
classifier.fit(train_data, y_train)

# 在测试数据上进行预测
y_pred = classifier.predict(test_data)
y_pred_spam = classifier.predict(spam_data)
y_pred_ham = classifier.predict(ham_data)

# 计算分类器的性能指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy_spam = accuracy_score(spam_data['label'], y_pred_spam)
precision_spam = precision_score(spam_data['label'], y_pred_spam)
recall_spam = recall_score(spam_data['label'], y_pred_spam)
f1_spam = f1_score(spam_data['label'], y_pred_spam)
accuracy_ham = accuracy_score(ham_data['label'], y_pred_ham)
precision_ham = precision_score(ham_data['label'], y_pred_ham)
recall_ham = recall_score(ham_data['label'], y_pred_ham)
f1_ham = f1_score(ham_data['label'], y_pred_ham)

# 输出性能指标
print("准确率:", accuracy)
print("精确率:", precision)
print("召回率:", recall)
print("F1值:", f1)
