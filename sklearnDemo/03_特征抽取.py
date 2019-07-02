# -*- coding: utf-8 -*-
# @createTime    : 2019/7/1 22:31
# @author  : Huanglg
# @fileName: 03_特征抽取.py
# @email: luguang.huang@mabotech.com

from sklearn.feature_extraction import DictVectorizer

def dict_demo():
    """对字典数据进行特征抽取"""

    data = [{'city': '北京','temperature':100}, {'city': '上海','temperature':60}, {'city': '深圳','temperature':30}]
    # 1.实例化一个转换器
    transfer = DictVectorizer(sparse=True)
    # 2.调用fit_transform
    data = transfer.fit_transform(data)
    print("返回的结果:\n", data)
    print("特征名字", transfer.get_feature_names())


if __name__ == '__main__':
    dict_demo()

