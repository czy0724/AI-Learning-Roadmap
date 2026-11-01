"""
案例:
    演示如何创建全0，全1，指定值的张量.

涉及到的函数如下:
    torch.ones 和 torch.ones_like 创建全1张量
    torch.zeros 和 torch.zeros_like 创建全0张量
    torch.full 和 torch.full_like 创建全为指定值张量

需要你掌握的函数:
    zeros(), full()
"""

# 导包
import torch

# 场景1: torch.ones 和 torch.ones_like 创建全1张量
t1 = torch.ones(size=(2, 3)) # 2行3列的全1张量
print(t1)
print('-' * 30)

t2 = torch.ones_like(t1) # 2行3列的全1张量，和t1形状一样(不是数值一样)
print(t2)
print('-' * 30)

# 场景2: torch.zeros 和 torch.zeros_like 创建全0张量
t3 = torch.zeros(size=(2, 3)) # 2行3列的全0张量
print(t3)
print('-' * 30)

t4 = torch.zeros_like(t1) # 2行3列的全0张量，和t1形状一样(不是数值一样)
print(t4)
print('-' * 30)

# 场景3: torch.full 和 torch.full_like 创建全为指定值张量
t5 = torch.full(size=(2, 3), fill_value=5) # 2行3列的全5张量
print(t5)
print('-' * 30)

t6 = torch.full_like(t1, fill_value=5) # 2行3列的全5张量，和t1形状一样(不是数值一样)
print(t6)
print('-' * 30)

