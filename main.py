from collections import deque
import os
import random
from tqdm import tqdm

import torch

from utils_drl import Agent
from utils_env import MyEnv
from utils_memory import ReplayMemory


GAMMA = 0.99                #  γ(Discount Factor)，表示非立即奖励(future rewards)相对于立即奖励(present rewards)的重要程度
GLOBAL_SEED = 0
MEM_SIZE = 100_000
RENDER = False
SAVE_PREFIX = "./models_D_v2"
STACK_SIZE = 4

EPS_START = 1.              # ε初始值,1代表开始时完全随机
EPS_END = 0.1               # ε最小值
EPS_DECAY = 1000000

BATCH_SIZE = 32             # 每次在memory中提取出的字节?数,一批数量的大小
POLICY_UPDATE = 4           # 进行learn的周期
TARGET_UPDATE = 10_000      # 进行sync周期
WARM_STEPS = 50_000         # 设定的内存上限
MAX_STEPS = 2_500_000       # 最大轮数
EVALUATE_FREQ = 100_000     # 重置环境并进行输出,标志下一轮ep的开始的 周期

rand = random.Random()
rand.seed(GLOBAL_SEED)
new_seed = lambda: rand.randint(0, 1000_000)
os.mkdir(SAVE_PREFIX)

torch.manual_seed(new_seed())
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# 获取环境
env = MyEnv(device)
# 定义Agent
agent = Agent(
    env.get_action_dim(),
    device,
    GAMMA,
    new_seed(),
    EPS_START,
    EPS_END,
    EPS_DECAY,
)
memory = ReplayMemory(STACK_SIZE + 1, MEM_SIZE, device)

# 开始训练
obs_queue: deque = deque(maxlen=5)
done = True

# Tqdm 是一个快速，可扩展的Python进度条  步数,宽度,留空,单位
progressive = tqdm(range(MAX_STEPS), total=MAX_STEPS,
                   ncols=50, leave=False, unit="b")
for step in progressive:
    if done:
        # reset重置环境为初始状态，并且返回该初始状态；
        observations, _, _ = env.reset()
        for obs in observations:
            obs_queue.append(obs)

    # 记忆集最大数量
    training = len(memory) > WARM_STEPS
    # 生成当前state
    state = env.make_state(obs_queue).to(device).float()
    # 获得一个操作 (ε-greedy等)
    action = agent.run(state, training)
    # Step将一个操作转发给环境，并返回最新的观察结果、奖励和指示训练是否结束的bool值。
    # 对象,价值,完成标志
    obs, reward, done = env.step(action)
    obs_queue.append(obs)
    memory.push(env.make_folded_state(obs_queue), action, reward, done)

    if step % POLICY_UPDATE == 0 and training:
        # 训练价值网络(通过时间差分-learning)
        loss = agent.learn(memory, BATCH_SIZE)
        # 本波最后一次训练写入 训练次数 , step, loss
        if step % EVALUATE_FREQ == 0:
            with open("rewards_D_v2.txt", "a") as fp:
                fp.write(f"LOSS {step // POLICY_UPDATE:3d} {step:8d} {loss}\n")

    if step % TARGET_UPDATE == 0:
        # 更新一些target网络的参数
        agent.sync()

    if step % EVALUATE_FREQ == 0:
        # 使用给定的代理运行游戏并返回平均reward和捕获的帧
        avg_reward, frames = env.evaluate(obs_queue, agent, render=RENDER)
        # 写入 ( 第n波 step 平均reward)
        with open("rewards_D_v2.txt", "a") as fp:
            fp.write(
                f"REWD {step // EVALUATE_FREQ:3d} {step:8d} {avg_reward:.1f}\n")
        if RENDER:
            prefix = f"eval_{step//EVALUATE_FREQ:03d}"
            os.mkdir(prefix)
            for ind, frame in enumerate(frames):
                with open(os.path.join(prefix, f"{ind:06d}.png"), "wb") as fp:
                    frame.save(fp, format="png")
        # 保存网络中的参数,到model_波数中
        agent.save(os.path.join(
            SAVE_PREFIX, f"model_{step//EVALUATE_FREQ:03d}"))
        done = True
