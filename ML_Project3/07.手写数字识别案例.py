"""
案例: 演示 KNN算法 识别图片, 即: 手写数字识别案例.

介绍:
    每张图片都是由 28 * 28 像素组成的, 即: 我们的csv文件中每一行都有 784个像素点, 表示图片(每个像素)的 颜色.
    最终构成图像.
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
from collections import Counter
#忽略警告
import warnings
warnings.filterwarnings('ignore', module='sklearn') #参一:忽略警告, 参二:忽略的模块名


#1.定义函数,接受用户传入的索引,展示该索引对应的图片
def show_digit(idx):
    #1.读取数据集,获取原数据
    df = pd.read_csv('./data/手写数字识别.csv')

    #2.判断传入的索引是否越界
    if idx < 0 or idx > len(df)-1:
        print('索引越界')
        return

    #3.走这里,说明没越界,就正常获取数据
    x = df.iloc[:,1:]
    y = df.iloc[:,0]

    #4.查看用户传入的索引对应的图片
    print(f'该图片对应的数字是: {y.iloc[idx]}')

    #5.查看下 查看用户传入的索引对应的图片 的形状
    print(x.iloc[idx].shape)


    #6.把784个像素点,重新 reshape成 28 * 28的形状
    x = x.iloc[idx].values.reshape(28,28)

    #7.具体的绘制灰度图的动作
    plt.imshow(x, cmap='gray')
    plt.axis('off')#关闭坐标轴
    plt.show()

#2.定义函数,训练模型,并保存训练好的模型
def train_model():
    #1.加载数据集
    df = pd.read_csv('./data/手写数字识别.csv')
    #2.数据预处理
    #2.1拆分出特征列
    x = df.iloc[:,1:]
    #2.2拆分出目标列
    y = df.iloc[:,0]
    #2.3打印特征和标签形状
    print(f'x的形状: {x.shape}')  # (42000, 784)
    print(f'y的形状: {y.shape}')  # (42000,)
    print(f'查看所有的标签的分布情况: {Counter(y)}')
    # 2.4 对特征列(拆分前)进行 归一化.
    x = x / 255
    # 2.5 拆分训练集和测试集.
    # 参数1: 特征列, 参数2: 标签列, 参数3: 测试集的比例, 参数4: 随机种子, 参数5: 参考y值抽取,保持标签比例(数据均衡)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20, stratify=y)

    #3.模型训练
    #3.1创建模型
    estimator = KNeighborsClassifier(n_neighbors=3)
    #3.2训练模型
    estimator.fit(x_train, y_train)

    #4.模型评估
    print(f'模型准确率: {estimator.score(x_test, y_test)}')
    print(f'模型准确率: {accuracy_score(y_test, estimator.predict(x_test))}')

    #5.保存模型
    #参一: 模型对象, 参二: 保存路径
    joblib.dump(estimator, './model/手写数字识别.pkl')
    print('模型保存成功')



#3定义函数,测试模型
def use_model():
    #1.加载图片
    img = plt.imread('./data/demo.png') #
    #2.绘制图片
    # plt.imshow(img, cmap='gray')
    # plt.axis('off')
    # plt.show()

    #3.加载模型
    estimator = joblib.load('./model/手写数字识别.pkl')

    #4.模型预测
    #4.1查看数据集转换
    print(img.reshape(1,-1).shape) #效果等同于(1, 784)

    #4.2 具体的转换动作,记得: 归一化(因为训练模型的时候使用了归一化动作)
    #x = img.reshape(1,-1)/255  #模型可能会预测失败，因为读图的时候，像素值可能不是特别的准确。
    x = img.reshape(1,-1)       #用原始的读取到的像素值，做预测。

    #4.3模型预测
    y_pre = estimator.predict(x)

    #5打印预测结果
    print(f'预测结果是: {y_pre}')










#4.测试
if __name__ == '__main__':
    #绘制数字
     # show_digit(9)
    #训练模型并保存模型
    # train_model()
    #模型预测
    use_model()