import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt

# 假设环境保存在 env.py 中
from env import MultiUAVTrackingEnv, Config as EnvConfig


# ==========================================
# 1. MAPPO 算法超参数配置
# ==========================================
class MAPPOConfig:
    # 网络架构
    HIDDEN_DIM = 128

    # 训练超参数
    TOTAL_TIMESTEPS = 1_000_000  # 总训练环境交互步数
    NUM_STEPS = 2000  # 每次收集轨迹的步数 (Rollout length)
    MINIBATCH_SIZE = 250  # 每次网络更新的批次大小
    UPDATE_EPOCHS = 10  # 每次收集后网络更新的轮数

    # PPO 核心参数
    LR = 3e-4  # 初始学习率
    GAMMA = 0.99  # 折扣因子
    GAE_LAMBDA = 0.95  # GAE 优势估计衰减
    CLIP_COEF = 0.2  # PPO 截断范围
    ENT_COEF = 0.01  # 熵增益系数 (鼓励探索)
    VF_COEF = 0.5  # 价值函数损失系数
    MAX_GRAD_NORM = 0.5  # 梯度裁剪阈值

    # 探索率 (动作标准差) 线性衰减
    ACTION_STD_INIT = 0.5  # 初始标准差
    ACTION_STD_FINAL = 0.05  # 最终标准差

    # 设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. 神经网络结构 (Actor & Centralized Critic)
# ==========================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """PPO 标配：正交初始化"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)  # 输出层 std 小一点，动作更平滑
        )
        # 动作标准差不作为网络参数学习，而是外部控制衰减
        self.action_std = MAPPOConfig.ACTION_STD_INIT

    def forward(self, obs):
        action_mean = torch.tanh(self.net(obs))  # 将均值约束在 [-1, 1]
        return action_mean

    def get_action(self, obs, action=None):
        action_mean = self.forward(obs)
        cov_mat = torch.diag(torch.full((action_mean.shape[-1],), self.action_std ** 2)).to(MAPPOConfig.DEVICE)
        dist = Normal(action_mean, torch.sqrt(torch.diag(cov_mat)))

        if action is None:
            action = dist.sample()
            action = torch.clamp(action, -1.0, 1.0)  # 环境要求限制在 [-1, 1]

        return action, dist.log_prob(action).sum(dim=-1), dist.entropy().sum(dim=-1)


class Critic(nn.Module):
    def __init__(self, global_obs_dim, hidden_dim):
        """Critic 拥有上帝视角，输入维度是所有智能体局部观测的拼接"""
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(global_obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        )

    def forward(self, global_obs):
        return self.net(global_obs)


# ==========================================
# 3. MAPPO 智能体类
# ==========================================
class MAPPOAgent:
    def __init__(self, num_agents, obs_dim, action_dim, cfg):
        self.cfg = cfg
        self.num_agents = num_agents

        # 全局状态维度：所有无人机的观测拼接在一起
        global_obs_dim = obs_dim * num_agents

        # 初始化共享网络
        self.actor = Actor(obs_dim, action_dim, cfg.HIDDEN_DIM).to(cfg.DEVICE)
        self.critic = Critic(global_obs_dim, cfg.HIDDEN_DIM).to(cfg.DEVICE)

        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': cfg.LR},
            {'params': self.critic.parameters(), 'lr': cfg.LR}
        ], eps=1e-5)

    def set_decay(self, progress):
        """更新探索率(动作标准差)和学习率，实现线性衰减"""
        # 学习率线性衰减
        lr = self.cfg.LR * (1.0 - progress)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # 标准差线性衰减
        self.actor.action_std = self.cfg.ACTION_STD_INIT - progress * (
                    self.cfg.ACTION_STD_INIT - self.cfg.ACTION_STD_FINAL)

    def update(self, rollouts):
        """计算损失并更新网络"""
        b_obs, b_global_obs, b_actions, b_logprobs, b_returns, b_advantages = rollouts

        # 展平批次和智能体维度，因为我们使用的是参数共享，所有智能体数据混在一起更新 Actor
        # 但要注意，Critic 的输入是拼接后的 global_obs
        batch_size = b_obs.shape[0]

        b_inds = np.arange(batch_size)
        clipfracs = []

        for epoch in range(self.cfg.UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, self.cfg.MINIBATCH_SIZE):
                end = start + self.cfg.MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                # 拿出一小批数据
                mb_obs = b_obs[mb_inds]  # [minibatch, num_agents, obs_dim]
                mb_global_obs = b_global_obs[mb_inds]  # [minibatch, global_obs_dim]
                mb_actions = b_actions[mb_inds]  # [minibatch, num_agents, action_dim]
                mb_logprobs = b_logprobs[mb_inds]  # [minibatch, num_agents]
                mb_advantages = b_advantages[mb_inds]  # [minibatch, num_agents]
                mb_returns = b_returns[mb_inds]  # [minibatch, num_agents]

                # --- 核心计算 ---
                # Actor 评估 (展平 multi-agent 维度一起送进去)
                flat_mb_obs = mb_obs.view(-1, mb_obs.shape[-1])
                flat_mb_actions = mb_actions.view(-1, mb_actions.shape[-1])
                _, newlogprob, entropy = self.actor.get_action(flat_mb_obs, flat_mb_actions)

                # 恢复形状
                newlogprob = newlogprob.view(-1, self.num_agents)
                entropy = entropy.view(-1, self.num_agents)

                # Critic 评估 (输入 global_obs，输出的 shape [minibatch, 1]，扩展为 [minibatch, num_agents])
                newvalue = self.critic(mb_global_obs)
                newvalue = newvalue.expand(-1, self.num_agents)

                # --- Actor 损失 ---
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    clipfracs += [((ratio - 1.0).abs() > self.cfg.CLIP_COEF).float().mean().item()]

                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.cfg.CLIP_COEF, 1 + self.cfg.CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # --- Critic 损失 ---
                v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                # --- 熵增益 (鼓励探索) ---
                entropy_loss = entropy.mean()

                # --- 总体损失与反向传播 ---
                loss = pg_loss - self.cfg.ENT_COEF * entropy_loss + v_loss * self.cfg.VF_COEF

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.MAX_GRAD_NORM)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.MAX_GRAD_NORM)
                self.optimizer.step()


# ==========================================
# 4. 数据记录与绘图类
# ==========================================
class DataLogger:
    def __init__(self, save_dir="MAPPO_Results"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.data_list = []

    def append(self, ep_data):
        self.data_list.append(ep_data)

    def save_and_plot(self, window_size=50):
        if not self.data_list: return

        df = pd.DataFrame(self.data_list)
        csv_path = os.path.join(self.save_dir, "training_log.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n[*] 训练数据已保存至: {csv_path}")

        # 绘制平滑曲线
        metrics_to_plot = ['ep_reward', 'ep_length', 'track_ratio', 'success_rate']
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, metric in enumerate(metrics_to_plot):
            if metric not in df.columns: continue
            ax = axes[i]
            y_raw = df[metric].values

            # 计算滑动平均
            y_smooth = pd.Series(y_raw).rolling(window=window_size, min_periods=1).mean().values

            ax.plot(df['step'], y_raw, alpha=0.3, color='gray', label='Raw')
            ax.plot(df['step'], y_smooth, color='red', linewidth=2, label=f'Smoothed (w={window_size})')
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xlabel("Timesteps")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        img_path = os.path.join(self.save_dir, "training_trends.png")
        plt.savefig(img_path, dpi=300)
        print(f"[*] 训练趋势图已保存至: {img_path}")
        plt.close()


# ==========================================
# 5. 主训练逻辑
# ==========================================
def train():
    env = MultiUAVTrackingEnv(EnvConfig())
    cfg = MAPPOConfig()

    obs_dim = env.observation_space.shape[1]
    action_dim = env.action_space.shape[1]
    num_agents = env.cfg.NUM_AGENTS

    agent = MAPPOAgent(num_agents, obs_dim, action_dim, cfg)
    logger = DataLogger()

    # 存储轨迹的 Tensor
    obs = torch.zeros((cfg.NUM_STEPS, num_agents, obs_dim)).to(cfg.DEVICE)
    global_obs = torch.zeros((cfg.NUM_STEPS, num_agents * obs_dim)).to(cfg.DEVICE)
    actions = torch.zeros((cfg.NUM_STEPS, num_agents, action_dim)).to(cfg.DEVICE)
    logprobs = torch.zeros((cfg.NUM_STEPS, num_agents)).to(cfg.DEVICE)
    rewards = torch.zeros((cfg.NUM_STEPS, num_agents)).to(cfg.DEVICE)
    dones = torch.zeros((cfg.NUM_STEPS, num_agents)).to(cfg.DEVICE)
    values = torch.zeros((cfg.NUM_STEPS, num_agents)).to(cfg.DEVICE)

    global_step = 0
    ep_rewards = []
    ep_lengths = []
    ep_track_steps = 0
    ep_lose_steps = 0
    success_buffer = []  # 记录最近100局的成功率

    next_obs, _ = env.reset()
    next_obs = torch.Tensor(next_obs).to(cfg.DEVICE)
    next_done = torch.zeros(num_agents).to(cfg.DEVICE)

    num_updates = cfg.TOTAL_TIMESTEPS // cfg.NUM_STEPS

    print("=========================================================")
    print(f"🚀 开始 MAPPO 训练 | 设备: {cfg.DEVICE} | 总步数: {cfg.TOTAL_TIMESTEPS}")
    print("=========================================================")

    for update in range(1, num_updates + 1):
        # 计算进度，更新学习率和探索率
        progress = (update - 1.0) / num_updates
        agent.set_decay(progress)

        ep_returns_batch = []

        # 收集数据 (Rollout)
        for step in range(0, cfg.NUM_STEPS):
            global_step += 1

            obs[step] = next_obs
            dones[step] = next_done

            # 构建 Centralized Critic 需要的全局观测 (直接把所有智能体的obs拼起来)
            g_obs = next_obs.view(-1)
            global_obs[step] = g_obs

            with torch.no_grad():
                action, logprob, _ = agent.actor.get_action(next_obs)
                # Critic 评估全局状态，得到 [1]，扩展给每个智能体
                val = agent.critic(g_obs)
                values[step] = val.expand(num_agents)

            actions[step] = action
            logprobs[step] = logprob

            # 与环境交互
            action_np = action.cpu().numpy()
            next_obs_np, reward_np, terminated, truncated, info = env.step(action_np)

            rewards[step] = torch.tensor(reward_np).to(cfg.DEVICE).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(cfg.DEVICE)

            # 统计业务指标
            ep_rewards.append(np.mean(reward_np))

            # 统计跟踪状态: 如果任一无人机看到目标，算作跟踪时间
            # (这里利用 env 中状态空间的视距标记位。我们知道 obs 的索引，或者直接用 info，为了稳妥用环境步数简单估算)
            if info.get('reason') != 'target_lost' and not terminated:
                # 假设正常步进且未丢失，简单按奖励是否扣除 lose_step 算
                # 最好在 env step 里把 track_flag 传出来。这里假设只要惩罚里没扣 lose，就是 track
                ep_track_steps += 1

            if terminated or truncated:
                ep_length = len(ep_rewards)
                total_ep_reward = sum(ep_rewards)
                is_success = info.get('is_success', 0)

                ep_lengths.append(ep_length)
                success_buffer.append(is_success)
                if len(success_buffer) > 100: success_buffer.pop(0)

                # 记录这一局的数据
                logger.append({
                    'step': global_step,
                    'ep_reward': total_ep_reward,
                    'ep_length': ep_length,
                    'track_ratio': ep_track_steps / ep_length,  # 跟踪时长占比
                    'success_rate': np.mean(success_buffer)
                })

                ep_rewards = []
                ep_track_steps = 0
                ep_lose_steps = 0
                next_obs, _ = env.reset()
                next_obs = torch.Tensor(next_obs).to(cfg.DEVICE)
                next_done = torch.zeros(num_agents).to(cfg.DEVICE)

            next_done = torch.Tensor([terminated or truncated] * num_agents).to(cfg.DEVICE)

        # 优势计算 (GAE)
        with torch.no_grad():
            next_value = agent.critic(next_obs.view(-1)).expand(num_agents)
            advantages = torch.zeros_like(rewards).to(cfg.DEVICE)
            lastgaelam = 0
            for t in reversed(range(cfg.NUM_STEPS)):
                if t == cfg.NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + cfg.GAMMA * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + cfg.GAMMA * cfg.GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + values

        # 网络更新
        agent.update((obs, global_obs, actions, logprobs, returns, advantages))

        # 定期打印调试信息
        if update % 5 == 0:
            df = pd.DataFrame(logger.data_list) if logger.data_list else None
            avg_rew = np.mean(df['ep_reward'].values[-10:]) if logger.data_list else 0
            avg_len = np.mean(df['ep_length'].values[-10:]) if logger.data_list else 0
            succ_rate = np.mean(success_buffer) if success_buffer else 0
            curr_lr = agent.optimizer.param_groups[0]['lr']
            curr_std = agent.actor.action_std

            print(f"| Update: {update:03d}/{num_updates} | Step: {global_step:07d} "
                  f"| Avg Reward: {avg_rew:>7.2f} | Avg Len: {avg_len:>5.1f} | Succ Rate: {succ_rate:>5.2f} "
                  f"| LR: {curr_lr:.2e} | Std: {curr_std:.3f} |")

    # 训练结束保存数据和模型
    print("=========================================================")
    print("🎉 训练完成，正在保存数据与模型...")
    logger.save_and_plot()

    torch.save(agent.actor.state_dict(), os.path.join(logger.save_dir, "mappo_actor.pth"))
    torch.save(agent.critic.state_dict(), os.path.join(logger.save_dir, "mappo_critic.pth"))
    print("[*] 模型权重已保存至: mappo_actor.pth 和 mappo_critic.pth")


if __name__ == "__main__":
    train()