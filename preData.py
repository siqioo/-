from email.parser import Parser
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

filetype="utf-8"
# 解析邮件内容
def get_body(msg):
    if msg.is_multipart():
        return get_body(msg.get_payload(0))
    else:
        return msg.get_payload(None, decode=True)

if __name__ == '__main__':
    f = open("./trec06p/full/index", encoding=filetype)
    f1 = open("data.txt", "a+", encoding=filetype, errors='ignore')
    line = f.readline()
    num = 0
    while line:
        line = line.rstrip()
        temp = line.split(' ', 1)  # 以空格为分隔符，分隔成两个
        print(num / 37822 * 100)
        num += 1
        print("目前文件数:", num)
        path = ".\\trec06p" + temp[1].lstrip('..').replace('/', "\\")
        with open(path, encoding=filetype, errors='ignore') as f2:
            text = f2.read()
        f2.close()
        email = Parser().parsestr(text)
        text=get_body(email).decode(filetype, errors='ignore').replace('\n', '').replace('\r', '').replace('\t', '').strip()
        text=temp[0]+'\t'+text+'\n'
        f1.write(text)
        print(text)
        line = f.readline()
    f.close()
    f1.close()
    print("处理完成")

# 下载停用词数据
nltk.download('stopwords')
# 加载停用词
stop_words = set(stopwords.words('english'))

# 读取垃圾邮件数据
def preprocess_data(data_file):
    data_init = pd.read_table(data_file, sep='\t', names=['label', 'mem'])
    # 数据预处理
    data_init['label'] = data_init.label.map({'ham': 0, 'spam': 1})  # 0代表正常邮件，1代表垃圾邮件
    total_count = data_init.shape[0]
    spam_count = np.count_nonzero(data_init['label'].values)  # 垃圾邮件数目
    print("邮件数目:", total_count)
    print("垃圾邮件数目:", spam_count)
    print("正常邮件数目:", total_count - spam_count)

    # 划分测试集和训练集
    x_train, x_test, y_train, y_test = train_test_split(data_init['mem'], data_init['label'], random_state=1,
                                                        stratify=data_init['label'])

    # 填充缺失值
    x_train = x_train.fillna("")
    x_test = x_test.fillna("")
    print('训练集大小: {}'.format(x_train.shape[0]))
    print('测试集大小: {}'.format(x_test.shape[0]))

    return x_train, x_test, y_train, y_test
