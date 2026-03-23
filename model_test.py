import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Wedge
from matplotlib.animation import FuncAnimation, PillowWriter

# 导入环境和网络配置
from env import MultiUAVTrackingEnv, Config as EnvConfig
from MAPPO import Actor, MAPPOConfig


def generate_episode_data(env, actor, device):
    """
    预演一局完整的环境，收集所有帧的状态数据，以便流畅渲染
    """
    obs, _ = env.reset()
    done = False
    step = 0
    total_reward = 0

    history = []
    static_obs = env.static_obs.copy()

    while not done and step < env.cfg.MAX_STEPS:
        # 记录当前帧状态
        history.append({
            'uav_pos': np.copy(env.uav_pos),
            'uav_yaw': np.copy(env.uav_yaw),
            'target_pos': np.copy(env.target_pos),
            'reward': total_reward,
            'step': step
        })

        # --- 核心：测试阶段使用确定性策略 ---
        # 直接调用 forward 获取 action_mean，而不是 get_action() 采样
        obs_tensor = torch.FloatTensor(obs).to(device)
        with torch.no_grad():
            action = actor(obs_tensor).cpu().numpy()

        # 与环境交互
        obs, rewards, terminated, truncated, info = env.step(action)
        total_reward += np.mean(rewards)  # 记录双机平均得分
        done = terminated or truncated
        step += 1

    # 记录结束帧
    history.append({
        'uav_pos': np.copy(env.uav_pos),
        'uav_yaw': np.copy(env.uav_yaw),
        'target_pos': np.copy(env.target_pos),
        'reward': total_reward,
        'step': step,
        'reason': info.get('reason', '未知')
    })

    return history, static_obs


def render_beautiful_gif(history, static_obs, env_cfg, filename):
    """
    将收集到的单局数据渲染为高质量美观的 GIF
    """
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)

    # 科技感配色方案
    COLOR_OBS = '#34495E'  # 深蓝灰 (障碍物)
    COLOR_UAV1 = '#2980B9'  # 克莱因蓝 (无人机1)
    COLOR_UAV2 = '#27AE60'  # 祖母绿 (无人机2)
    COLOR_TARGET = '#E74C3C'  # 珊瑚红 (目标点)
    COLOR_BG = '#F8F9F9'  # 极浅灰背景

    def update(frame_idx):
        ax.clear()
        data = history[frame_idx]

        # 1. 基础画布设置
        ax.set_xlim(0, env_cfg.MAP_SIZE)
        ax.set_ylim(0, env_cfg.MAP_SIZE)
        ax.set_facecolor(COLOR_BG)
        ax.grid(color='white', linestyle='-', linewidth=1.5)

        # 隐藏边框
        for spine in ax.spines.values():
            spine.set_visible(False)

        # 标题显示当前状态
        reason_str = f" | End Reason: {data.get('reason')}" if 'reason' in data else ""
        ax.set_title(f"Step: {data['step']:03d} | Avg Reward: {data['reward']:.2f}{reason_str}",
                     fontsize=14, fontweight='bold', pad=15)

        # 2. 绘制静态障碍物
        for obs in static_obs:
            if obs['shape'] == 'circle':
                circle = Circle(obs['pos'], obs['size'], color=COLOR_OBS, alpha=0.8, ec='white', lw=1.5)
                ax.add_patch(circle)
            else:
                rect = Rectangle(obs['pos'] - obs['size'], obs['size'] * 2, obs['size'] * 2,
                                 color=COLOR_OBS, alpha=0.8, ec='white', lw=1.5)
                ax.add_patch(rect)

        # 3. 绘制历史轨迹线 (Tails)
        if frame_idx > 0:
            hist_target = np.array([h['target_pos'] for h in history[:frame_idx + 1]])
            hist_uav1 = np.array([h['uav_pos'][0] for h in history[:frame_idx + 1]])
            hist_uav2 = np.array([h['uav_pos'][1] for h in history[:frame_idx + 1]])

            ax.plot(hist_target[:, 0], hist_target[:, 1], color=COLOR_TARGET, linestyle=':', lw=2, alpha=0.5)
            ax.plot(hist_uav1[:, 0], hist_uav1[:, 1], color=COLOR_UAV1, linestyle='--', lw=2, alpha=0.4)
            ax.plot(hist_uav2[:, 0], hist_uav2[:, 1], color=COLOR_UAV2, linestyle='--', lw=2, alpha=0.4)

        # 4. 绘制目标
        target_pos = data['target_pos']
        ax.plot(target_pos[0], target_pos[1], marker='*', color=COLOR_TARGET, markersize=16,
                markeredgecolor='white', markeredgewidth=1.0, label='Target', zorder=5)

        # 5. 绘制无人机及其探测扇形 (FOV)
        uav_colors = [COLOR_UAV1, COLOR_UAV2]
        for i in range(env_cfg.NUM_AGENTS):
            pos = data['uav_pos'][i]
            yaw = data['uav_yaw'][i]

            # 画无人机本体 (带白边更立体)
            ax.plot(pos[0], pos[1], marker='o', color=uav_colors[i], markersize=10,
                    markeredgecolor='white', markeredgewidth=1.5, label=f'UAV {i + 1}', zorder=5)

            # 画探测扇形 (Wedge 的角度参数为角度制)
            theta1 = np.degrees(yaw - env_cfg.FOV / 2)
            theta2 = np.degrees(yaw + env_cfg.FOV / 2)
            wedge = Wedge(pos, env_cfg.R_SENSE, theta1, theta2,
                          color=uav_colors[i], alpha=0.15, zorder=2)
            ax.add_patch(wedge)

            # 画机头朝向指示线
            nose_end = pos + np.array([np.cos(yaw), np.sin(yaw)]) * 3.0
            ax.plot([pos[0], nose_end[0]], [pos[1], nose_end[1]], color='white', lw=2, zorder=6)

        # 只在第一帧添加图例，防止闪烁
        if frame_idx == 0:
            ax.legend(loc='upper right', framealpha=0.9, edgecolor='none')

    # 生成动画
    print(f"[*] 正在渲染 {filename} (共 {len(history)} 帧)...")
    ani = FuncAnimation(fig, update, frames=len(history), repeat=False)

    # 保存为 GIF (15 FPS 视觉效果最流畅)
    ani.save(filename, writer=PillowWriter(fps=15))
    plt.close(fig)
    print(f"[+] 渲染完成: {filename}\n")


def main():
    # 1. 初始化设置
    cfg = MAPPOConfig()
    env_cfg = EnvConfig()
    env = MultiUAVTrackingEnv(env_cfg)

    # 获取维度信息
    obs_dim = env.observation_space.shape[1]
    action_dim = env.action_space.shape[1]

    # 2. 实例化 Actor 网络并加载权重
    actor = Actor(obs_dim, action_dim, cfg.HIDDEN_DIM).to(cfg.DEVICE)

    # === 已修改：直接从当前运行路径加载模型文件 ===
    model_path = "mappo_actor.pth"

    if not os.path.exists(model_path):
        print(f"❌ 找不到模型权重文件: {model_path}")
        print("请确保你已经将 mappo_actor.pth 放置在当前运行目录下。")
        return

    actor.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
    actor.eval()  # 切换到评估模式
    print(f"✅ 成功加载预训练模型: {model_path} (Device: {cfg.DEVICE})")

    # 3. 创建保存动图的文件夹
    save_dir = "MAPPO_Test_Gifs"
    os.makedirs(save_dir, exist_ok=True)

    # 4. 在 3 个随机环境中测试并渲染
    for i in range(3):
        print(f"--- 开始测试环境 {i + 1}/3 ---")
        # 预演获取数据
        history, static_obs = generate_episode_data(env, actor, cfg.DEVICE)

        # 渲染保存
        gif_filename = os.path.join(save_dir, f"UAV_Tracking_Test_{i + 1}.gif")
        render_beautiful_gif(history, static_obs, env_cfg, gif_filename)


if __name__ == "__main__":
    main()