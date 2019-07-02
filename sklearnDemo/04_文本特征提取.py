# -*- coding: utf-8 -*-
# @createTime    : 2019/7/2 22:42
# @author  : Huanglg
# @fileName: 04_文本特征提取.py
# @email: luguang.huang@mabotech.com
import jieba
from sklearn.feature_extraction.text import CountVectorizer

def text_count_demo():
    """
    对文本进行特征抽取
    :return:
    """
    data = ["life is short,i like like python", "life is too long,i dislike python"]
    data = ["人生苦短，我喜欢Python", "生活太长久，我不喜欢Python"]
    # 实例化一个转换类
    transfer = CountVectorizer()
    # 调用fit_transform
    data = transfer.fit_transform(data)
    print("文本特征抽取的结果\n", data.toarray())
    print("返回特征名字:\n", transfer.get_feature_names())

def cut_world(text):
    """
    对中文分词
    :param text:
    :return:
    """
    text = " ".join(list(jieba.cut(text)))
    return text

def text_chinese_count_demo2():
    """
    对中文提取
    :return:
    """
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    # 将原始数据转换成分好词的形式
    text_list = []
    for send in data:
        send = cut_world(send)
        text_list.append(send)

    # 实例化转换类
    transfer = CountVectorizer()
    data = transfer.fit_transform(text_list)
    print(data.toarray())
    print(transfer.get_feature_names())


if __name__ == '__main__':
    # text_count_demo()
    text_chinese_count_demo2()
