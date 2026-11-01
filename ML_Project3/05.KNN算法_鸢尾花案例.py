"""
案例：通过KNN算法实现 鸢尾花的 分类操作。

回顾：机器学习项目的研发流程
1. 加载数据。
2. 数据的预处理。
3. 特征工程(提取，预处理...)
4. 模型训练。
5. 模型评估。
6. 模型预测。
"""

#导包
from sklearn.datasets import load_iris        # 加载鸢尾花测试集的.
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV   # 分割训练集和测试集的
from sklearn.preprocessing import StandardScaler       # 数据标准化的
from sklearn.neighbors import KNeighborsClassifier     # KNN算法 分类对象
from sklearn.metrics import accuracy_score             # 模型评估的，计算模型预测的准确率

#1.定义函数,加载鸢尾花数据集,并查看数据集
def dm01_load_iris():
    #1.加载鸢尾花数据集
    iris_data = load_iris()
    #2.查看数据集
    # print(f"数据集,{iris_data}")
    #3.查看数据集所有的键
    print(f"数据集所有的键,{iris_data.keys()}")
    #4.查看数据集键对应的值
    print(f"数据集键对应的值,{iris_data.data[:5]}")  # 前5行数据
    print(f"具体的标签,{iris_data.target[:5]}")#前5行标签
    print(f"标签对应的名称,{iris_data.target_names}")#setosa:0,versicolor:1,virginica:2
    print(f"特征对应的名称,{iris_data.feature_names}")#sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)
    # print(f"数据集的描述,{iris_data.DESCR}")
    print(f"数据集的框架,{iris_data.frame}")#None
    print(f"数据集文件名,{iris_data.filename}")#iris.csv
    print(f"数据集的模型(在哪个包下),{iris_data.data_module}")#sklearn.datasets.data


#2.定义函数,绘制数据集散点图
def dm02_show_iris():
    #1.加载数据集
    iris_data = load_iris()
    #2.把鸢尾花数据集封装成DataFrame对象
    iris_df = pd.DataFrame(iris_data.data,columns=iris_data.feature_names)
    #3.给DataFrame对象添加标签列
    iris_df['label'] = iris_data.target
    # print(iris_df)

    #4.绘制数据集散点图
    #参数1:数据集,参数2:x轴特征,参数3:y轴特征,参数4:分组字段,参数5:是否显示拟合线
    sns.lmplot(data = iris_df,x = 'sepal length (cm)',y = 'sepal width (cm)',hue = 'label',fit_reg = False)

    #5.设置标题,显示
    plt.title('iris data')
    plt.tight_layout()#自动调整子图参数
    plt.show()

#3.定义函数,切分训练集和测试集
def dm03_split_train_test():
    #1.加载数据集
    iris_data = load_iris()
    #2.数据的预处理:从150行数据中,随机抽取20%的数据作为测试集,80%的数据作为训练集
    #参数1:特征数据,参数2:标签数据,参数3:测试集占比,参数4:随机种子(种子一致,结果一致)

    #返回训练集和测试集
    x_train,x_test,y_train,y_test = train_test_split(iris_data.data,iris_data.target,test_size=0.2,random_state=22)
    #3.打印切割后的结果
    print(f"训练集特征:{x_train},个数:{len(x_train)}")#120条,每条4列
    print(f"训练集标签:{y_train},个数:{len(y_train)}")#120条,每条1列
    print(f"测试集特征:{x_test},个数:{len(x_test)}")#30条,每条4列
    print(f"测试集标签:{y_test},个数:{len(y_test)}")#30条,每条1列



#4. 划分数据集，数据集一般分为训练集‑> 训练模型，提供给模型，做特征、做预测；测试集，评估模型效果。
def dm04_iris_evaluate_test():
     #1.加载数据集
     iris_data = load_iris()

     #2.数据预处理 从150行数据中,随机抽取20%的数据作为测试集,80%的数据作为训练集
     x_train,x_test,y_train,y_test = train_test_split(iris_data.data,iris_data.target,test_size=0.2,random_state=22)

     #3.特征工程
     # 思考1：特征提取：因为源数据只有4个特征列，且都是我们用的，所以这里无需做特征提取。
     # 思考2：特征预处理：因为源数据的4列特征差值不大，所以我们无需做特征预处理，但是，加入特征预处理会让模型预测效果更好，所以加入。
     #3.1创建标准化对象
     transfer = StandardScaler()
     #3.2对特征列进行标准化处理
     #fit_transform:兼具fit和transform功能,即：训练，转换。该函数适用于：第一次进行标准化的时候使用,一般用于处理训练集
     x_train = transfer.fit_transform(x_train)
     #transform只有转换,该函数适用于:重复进行标准化的时候使用,一般用于处理测试集
     x_test = transfer.transform(x_test)
     #4.模型训练
     #4.1创建模型对象
     estimator = KNeighborsClassifier(n_neighbors=3)
     #4.2具体的训练模型
     estimator.fit(x_train,y_train)

     #5.模型预测
     #场景一:对刚才切分的测试集(30条)进行测试
     #直接预测
     y_predict = estimator.predict(x_test)
     print(f"测试集预测结果:{y_predict}")

     #场景二:对新的数据(150条之外的)进行测试
     #自定义数据
     my_data=[[7.8,2.1,3.9,1.6]]
     #标准化处理
     my_data = transfer.transform(my_data)
     #预测
     my_predict = estimator.predict(my_data)
     print(f"自定义数据预测结果:{my_predict}")
     #查看上述数据集,每种分类的预测概率
     print(f"每种分类的预测概率:{estimator.predict_proba(my_data)}")

     #6.模型评估
     #方式一 直接评分:基于训练集的特征和训练集的标签
     print(f"正确率:{estimator.score(x_train,y_train)}")

     #方式二基于测试集的标签和预测结果进行评分
     print(f"正确率:{accuracy_score(y_test,y_predict)}")

#5.测试
if __name__ == '__main__':
    # dm01_load_iris()
    # dm02_show_iris()
    # dm03_split_train_test()
    dm04_iris_evaluate_test()






