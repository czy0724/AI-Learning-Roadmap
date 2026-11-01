"""
案例:
    演示GRU代码

GRU简介:
概述:
    Gated Recurrent Unit, 门控循环单元结构, 可以理解为是一个 简化版的智能管家,
    即能像LSTM那样处理长序列信息, 计算上也更简单, 更节省资源.
核心设计思路:
    重置门, 更新门.        就是在LSTM的基础上, 对部分门做合并.
优点:
    和LSTM一样, 既可以处理长序列语义关联, 能够有效缓解梯度消失和爆炸问题.
    关于长序列的处理效果: 优于传统RNN
    关于计算复杂度: 比LSTM要小.
缺点:
    不能完全解决梯度消失, 爆炸问题, 不能并行计算.
    在数据量和模型体量逐步增大的未来, 是RNN发展的关键瓶颈.
"""
# 导包
import torch
import torch.nn as nn

def dm_gru_for_base():
    # 1. 创建模型。
    # 参1: 词向量维度(输入维度)， 参2: 隐藏层维度(输出维度)， 参3: 隐藏层层数。
    gru = nn.GRU(5, 6, 1)

    # 2. 准备输入数据(即: 本次的输入)
    # 参1: 句子长度(sequence_length)， 参2: 批次大小(batch_size)， 参3: 词向量维度(即:输入维度 input_size)
    input = torch.randn(1, 3, 5)

    # 3. 初始化隐藏层(即: 上一时间步的隐藏状态)
    # 参1: 隐藏层的层数(即: 隐藏层层数 num_layers), 参2: 批次大小(batch_size), 参3: 隐藏层维度(即: 输出维度 hidden_size)
    h0 = torch.randn(1, 3, 6)

    # 4. 运行模型
    # 本次的输出, 本次的隐藏状态 = gru(本次的输入, 上一时刻的隐藏状态)
    output, hn = gru(input, h0)

    # 5. 打印结果.
    print(f'output: {output}, output.shape: {output.shape}')  # shape: (1, 3, 6)
    print(f'hidden: {hn}, hidden.shape: {hn.shape}')  # shape: (1, 3, 6)
    print(f'gru模型: {gru}')  # GRU(5, 6, 1)

# 测试代码
if __name__ == '__main__':
    dm_gru_for_base()
