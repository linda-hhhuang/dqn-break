import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    #     def __init__(self, action_dim, device):
    #         super(DQN, self).__init__()
    #         self.__conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
    #         self.__conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
    #         self.__conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
    #         self.__fc1 = nn.Linear(64*7*7, 512)
    #         self.__fc2 = nn.Linear(512, action_dim)
    #         self.__device = device

    #     def forward(self, x):
    #         x = x / 255.
    #         x = F.relu(self.__conv1(x))
    #         x = F.relu(self.__conv2(x))
    #         x = F.relu(self.__conv3(x))
    #         x = F.relu(self.__fc1(x.view(x.size(0), -1)))
    #         return self.__fc2(x)
    def __init__(self, action_dim, device):
        super(DQN, self).__init__()
        # 定义卷积和全连接层
        # 对由多个输入平面组成的输入信号进行二维卷积 输入4个矩阵(4帧画面) 得到32个特征 核为8*8 步为4
        self.__conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.__conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.__conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        # 全连接层
        # 卷积层输出64个特征map ,每个特征为7*7 输出512个神经元

        # 仅与价值有关,和状态无关
        self.__value1 = nn.Linear(64 * 7 * 7, 512)
        self.__value2 = nn.Linear(512, 1)

        # 即与价值有关,也和动作有关
        self.__advantage1 = nn.Linear(64 * 7 * 7, 512)
        self.__advantage2 = nn.Linear(512, action_dim)

        self.__device = device
        self.__action_dim = action_dim

    def forward(self, x):
        x = x / 255.
        x = F.relu(self.__conv1(x))
        x = F.relu(self.__conv2(x))
        x = F.relu(self.__conv3(x))

        x = x.view(x.size(0), -1)

        value = F.relu(self.__value1(x))
        value = self.__value2(value).expand(x.size(0), self.__action_dim)

        advantage = F.relu(self.__advantage1(x))
        advantage = self.__advantage2(advantage)

        # 结合,按作者意思用平均操作代替最大化
        x = value + \
            (advantage - advantage.mean(1).unsqueeze(1).expand(x.size(0), self.__action_dim))
        return x

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
