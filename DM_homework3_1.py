import glob

import re
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

# 数据路径
data_path = '20_newsgroups'

# 得到文本数据路径数据
classes_dir_path = glob.glob(data_path + '/*')
classes_txt_names = []
for item in classes_dir_path:
    classes_txt_names.append(glob.glob(item + '/*'))

# 读取数据
data = []
for c_txt_names in classes_txt_names:
    c_data = []
    for txt_name in c_txt_names:
        c_data.append(open(txt_name, errors='ignore').read())
    data.append(c_data)



def clean_text(text):
    # 缩写替换
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"cannot", "can not ", text)
    text = re.sub(r"what\'s", "what is", text)
    text = re.sub(r"What\'s", "what is", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve ", " have ", text)
    text = re.sub(r"n\'t", " not ", text)
    text = re.sub(r"i\'m", "i am ", text)
    text = re.sub(r"I\'m", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    # 单独的数字替换为英文
    text = re.sub(r" 1 ", " one ", text)
    text = re.sub(r" 2 ", " two ", text)
    text = re.sub(r" 3 ", " three ", text)
    text = re.sub(r" 4 ", " four ", text)
    text = re.sub(r" 5 ", " five ", text)
    text = re.sub(r" 6 ", " six ", text)
    text = re.sub(r" 7 ", " seven ", text)
    text = re.sub(r" 8 ", " eight ", text)
    text = re.sub(r" 9 ", " nine ", text)
    # 替换不可见字符以及各分隔符
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\\+', ' ', text)
    text = re.sub(r'/+', ' ', text)
    text = re.sub(r'_+', ' ', text)
    text = re.sub(r'--+', ' ', text)
    text = re.sub(r'\.', ' ', text)
    text = re.sub(r' +', ' ', text)

    return text

# 分词
def tokenize(text):
    token_words = word_tokenize(text)
    token_words = pos_tag(token_words)
    return token_words

# 去掉词性
def stem(token_words):
    wordnet_lematizer = WordNetLemmatizer()
    words_lematizer = []
    for word, tag in token_words:
        if tag.startswith('NN'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='n')  # n代表名词
        elif tag.startswith('VB'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='v')  # v代表动词
        elif tag.startswith('JJ'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='a')  # a代表形容词
        elif tag.startswith('R'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='r')  # r代表代词
        else:
            word_lematizer = wordnet_lematizer.lemmatize(word)
        words_lematizer.append(word_lematizer)
    return words_lematizer

# 去掉停用词
def delete_stopwords(token_words):
    """ 去停用词"""
    sr = stopwords.words('english')
    cleaned_words = [word for word in token_words if word not in sr]
    return cleaned_words

# 去掉数字
def is_number(s):
    """ 判断字符串是否为数字"""
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

# 删除特殊字符
def delete_characters(token_words):
    """去除特殊字符、数字"""
    characters = ['\'', "''", '``', ',', '.', ':', ';', '?', '(', ')', '[', ']', '&',
                  '!', '*', '@', '#', '$', '%', '-', '>', '<', '...', '^', '{', '}']
    words_list = [word for word in token_words if word not in characters and not is_number(word)]
    return words_list

# 全部转换为小写
def to_lower(token_words):
    words_lists = [x.lower() for x in token_words]
    return words_lists

# 文本预处理接口
def pre_process(text):
    text = clean_text(text)
    token_words = tokenize(text)
    token_words = stem(token_words)
    token_words = delete_stopwords(token_words)
    token_words = delete_characters(token_words)
    token_words = to_lower(token_words)
    return token_words

# 数据预处理
docs_feats = []  # 每个文档清洗后的字符串数据
words_set = set()  # 整个数据集的词集合
k = 1
for c_doc in data:
    print('\r', end='')
    print('{} / {}'.format(k, len(data)))
    for doc in c_doc:
        words = pre_process(doc)
        for word in words:
            words_set.add(word)
        docs_feats.append(' '.join(words))
    k += 1
print('数据集中出现的词共有 %d个' % len(words_set))
