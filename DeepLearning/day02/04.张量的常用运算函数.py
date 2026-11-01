"""
案例:
    演示张量常用的运算函数.
涉及到的 API(函数) 如下:
    sum(), max(), min(), mean()        → 都有 dim 参数, 0表示列, 1表示行
    pow(), sqrt(), exp(), log(), log2(), log10()   → 没有dim参数
掌握的函数:
    sum(), max(), min(), mean()
"""

#导包
import torch

# 1. 定义张量，记录初值.
t1 = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
], dtype=torch.float)
print(f't1: {t1}')

# 2. 演示 有dim参数的 函数.
print(t1.sum(dim=0))        # 按列求和
print(t1.sum(dim=1))        # 按行求和
print(t1.sum())             # 整体求和
print('-' * 30)

# max()求最大值, min()同理, 这里就不演示了.
print(t1.max(dim=0))        # 按列求最大值
print(t1.max(dim=1))        # 按行求最大值
print(t1.max())             # 整体求最大值
print('-' * 30)

# mean(), 计算平均值
print(t1.mean(dim=0))        # 按列求平均值
print(t1.mean(dim=1))        # 按行求平均值
print(t1.mean())             # 整体求平均值
print('-' * 30)

# 3. 演示 无dim参数的 函数.
print(t1.pow(2))             # 求平方
print(t1.sqrt())            # 求平方根
print(t1.exp())             # 求指数
print(t1.log())             # 求自然对数
print(t1.log2())            # 求以2为底的对数
print(t1.log10())           # 求以10为底的对数
print('-' * 30)






