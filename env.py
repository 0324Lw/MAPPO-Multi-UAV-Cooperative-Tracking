import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. 统一参数配置类 Config
# ==========================================
class Config:
    """
    双无人机协同跟踪动态目标环境配置
    """
    # 1.1 环境与动力学参数
    MAP_SIZE = 100.0  # 地图大小 100m x 100m
    DT = 0.1  # 仿真步长 (s)
    MAX_V = 5.0  # 无人机最大线速度 (m/s)
    MAX_W = np.pi / 2  # 无人机最大角速度 (rad/s)
    SAFE_DIST_UAV = 2.0  # 飞机间防撞安全距离 (m)

    # 1.2 传感器与目标参数
    R_SENSE = 15.0  # 传感器最大探测半径 (m)
    FOV = np.pi / 2  # 传感器视场角 (90度)
    TARGET_MAX_V = 4.0  # 目标最大移速 (m/s) (略低于无人机以便跟踪)
    TARGET_LOSE_LIMIT = 50  # 目标彻底丢失容忍步数 (50步 = 5秒)

    # 1.3 静态障碍物参数
    NUM_STATIC_OBS = 15  # 静态障碍物数量
    OBS_SIZE_MIN = 3.0
    OBS_SIZE_MAX = 5.0
    MIN_OBS_DIST = 5.0  # 障碍物最小间距

    # 1.4 训练超参数
    MAX_STEPS = 1000  # 每回合最大步数
    NUM_AGENTS = 2  # 智能体数量

    # 1.5 奖励函数系数
    COEFF_TRACK_BASE = 0.5  # 保持不变，鼓励视野内跟踪
    COEFF_STANDOFF = 0.5  # 保持不变，鼓励保持最佳距离
    COEFF_DISPERSION = 0.5  # 保持不变，后期协同的高级奖励
    COEFF_SMOOTH = -0.02  # 不要在初期束缚手脚
    COEFF_DANGER = -0.5  # 保持不变，软避障警示
    COEFF_LOSE_STEP = -0.05  # 把生存的痛苦降到最低
    COEFF_COLLISION = -20.0  # 保持不变，撞墙是大忌
    COEFF_LOSE_DONE = -10.0  # 原为 -20，稍微减轻跟丢的彻底失败感，鼓励多活几步
    COEFF_SUCCESS = 50.0  # 保持不变


# ==========================================
# 2. 强化学习环境类 MultiUAVTrackingEnv
# ==========================================
class MultiUAVTrackingEnv(gym.Env):
    def __init__(self, cfg=Config()):
        super(MultiUAVTrackingEnv, self).__init__()
        self.cfg = cfg

        # 动作空间：2架飞机，每架 [v, w] 归一化到 [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.cfg.NUM_AGENTS, 2), dtype=np.float32)

        # 状态空间：每架飞机 21 维特征
        # 目标相对位置(2) + 目标裕量(2) + 视野标志(2) + 队友状态(4) + 自身状态(2) + 3个障碍物(9) = 21维
        self.obs_dim = 21
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.cfg.NUM_AGENTS, self.obs_dim),
                                            dtype=np.float32)

        # 实体状态
        self.uav_pos = np.zeros((self.cfg.NUM_AGENTS, 2))
        self.uav_yaw = np.zeros(self.cfg.NUM_AGENTS)
        self.uav_v = np.zeros(self.cfg.NUM_AGENTS)
        self.uav_w = np.zeros(self.cfg.NUM_AGENTS)
        self.prev_actions = np.zeros((self.cfg.NUM_AGENTS, 2))

        self.target_pos = np.zeros(2)
        self.target_yaw = 0.0
        self.target_v = self.cfg.TARGET_MAX_V * 0.8
        self.lose_steps = 0

        self.static_obs = []
        self.step_count = 0

    def _get_boundary_dist(self, pos, obs):
        """计算质心到障碍物边界的最短距离"""
        if obs['shape'] == 'circle':
            dist = np.linalg.norm(pos - obs['pos']) - obs['size']
        else:
            dx = np.abs(pos[0] - obs['pos'][0]) - obs['size']
            dy = np.abs(pos[1] - obs['pos'][1]) - obs['size']
            if dx > 0 and dy > 0:
                dist = np.sqrt(dx ** 2 + dy ** 2)
            else:
                dist = max(dx, dy)
        return dist

    def _check_los(self, uav_idx):
        """视距检测：判断目标是否在无人机 i 的 FOV 内且无遮挡"""
        vec = self.target_pos - self.uav_pos[uav_idx]
        dist = np.linalg.norm(vec)
        if dist > self.cfg.R_SENSE:
            return False, dist, 0.0

        angle = np.arctan2(vec[1], vec[0])
        yaw_diff = (angle - self.uav_yaw[uav_idx] + np.pi) % (2 * np.pi) - np.pi
        if np.abs(yaw_diff) > self.cfg.FOV / 2.0:
            return False, dist, yaw_diff

        # 简化遮挡检测：检查连线是否穿过障碍物
        for obs in self.static_obs:
            d_to_obs = self._get_boundary_dist(self.uav_pos[uav_idx], obs)
            if d_to_obs < dist:  # 障碍物在目标前方，可能遮挡 (此处可换成射线检测，为保证效率用简化版)
                v1 = obs['pos'] - self.uav_pos[uav_idx]
                proj = np.dot(v1, vec / dist)
                if 0 < proj < dist:
                    perp_dist = np.linalg.norm(v1 - proj * (vec / dist))
                    if perp_dist < obs['size']:
                        return False, dist, yaw_diff
        return True, dist, yaw_diff

    def _generate_scenario(self):
        """生成静态障碍物并初始化位置，确保出生点绝对安全且无视线遮挡"""
        self.static_obs = []
        existing_positions = []

        # 1. 生成障碍物 (逻辑不变)
        for _ in range(self.cfg.NUM_STATIC_OBS):
            shape = 'circle' if np.random.rand() > 0.5 else 'square'
            size = np.random.uniform(self.cfg.OBS_SIZE_MIN, self.cfg.OBS_SIZE_MAX)
            for _ in range(50):
                pos = np.random.uniform(size, self.cfg.MAP_SIZE - size, 2)
                conflict = any(
                    np.linalg.norm(pos - ex_pos) < size + ex_size + self.cfg.MIN_OBS_DIST for ex_pos, ex_size in
                    existing_positions)
                if not conflict:
                    self.static_obs.append(
                        {'pos': pos, 'size': size, 'shape': shape, 'type': 0 if shape == 'circle' else 1})
                    existing_positions.append((pos, size))
                    break

        # 2. 带有重试机制的实体初始化
        max_init_retries = 200  # 最大重试次数，防止死循环
        for attempt in range(max_init_retries):
            # --- A. 尝试初始化无人机 ---
            center = np.random.uniform(15, self.cfg.MAP_SIZE - 15, 2)
            uav1 = center + np.random.uniform(-3, 3, 2)
            uav2 = center + np.random.uniform(-3, 3, 2)

            # 飞机间距不能太近
            if np.linalg.norm(uav1 - uav2) < self.cfg.SAFE_DIST_UAV:
                continue

            # [修改点1]：大幅增加无人机出生点与障碍物的安全裕度 (提高到 4.0 米)
            uav_safe = True
            for obs in self.static_obs:
                if self._get_boundary_dist(uav1, obs) < 4.0 or self._get_boundary_dist(uav2, obs) < 4.0:
                    uav_safe = False
                    break
            if not uav_safe:
                continue

            self.uav_pos[0], self.uav_pos[1] = uav1, uav2
            self.uav_yaw = np.random.uniform(-np.pi, np.pi, 2)

            # --- B. 尝试初始化目标点 ---
            dist = np.random.uniform(5.0, 10.0)
            # 让目标尽量生成在 UAV0 的正前方区域
            target_angle = self.uav_yaw[0] + np.random.uniform(-self.cfg.FOV / 4, self.cfg.FOV / 4)
            t_pos = self.uav_pos[0] + np.array([dist * np.cos(target_angle), dist * np.sin(target_angle)])

            # [修改点2]：检查目标是否越界 (留出 3.0 米的边界缓冲)
            if t_pos[0] < 3.0 or t_pos[0] > self.cfg.MAP_SIZE - 3.0 or t_pos[1] < 3.0 or t_pos[
                1] > self.cfg.MAP_SIZE - 3.0:
                continue

            # [修改点3]：严格检查目标出生点与障碍物的安全距离 (设定为 3.0 米)
            target_safe = True
            for obs in self.static_obs:
                if self._get_boundary_dist(t_pos, obs) < 3.0:
                    target_safe = False
                    break
            if not target_safe:
                continue

            self.target_pos = t_pos
            self.target_yaw = np.random.uniform(-np.pi, np.pi)

            # --- C. 视线 (LoS) 终极校验 ---
            # [修改点4]：不仅位置要安全，还必须保证开局瞬间 UAV0 与 Target 之间没有障碍物遮挡
            # 为了确保一定能看见，我们直接强行修正一下 UAV0 的机头朝向对准目标点附近
            self.uav_yaw[0] = target_angle
            seen, _, _ = self._check_los(0)

            if not seen:
                continue  # 如果居然被遮挡了，当前布局作废，重新生成！

            # 只要撑到了这一步，说明完美的出生点已经找到，跳出重试循环
            break
        else:
            # 如果循环了 200 次都没找到合适的位置（极端脸黑情况），递归重新生成整个地图
            self._generate_scenario()

    def _update_target(self):
        """目标平滑随机游走与被动避障"""
        # 基础游走 (添加OU噪声般的平滑转向)
        self.target_yaw += np.random.normal(0, 0.2)

        # 简单的势场避障
        repulsive_force = np.zeros(2)
        for obs in self.static_obs:
            d = self._get_boundary_dist(self.target_pos, obs)
            if d < 4.0:
                vec = self.target_pos - obs['pos']
                repulsive_force += (vec / (np.linalg.norm(vec) + 1e-5)) * (4.0 - d)

        # 边界避障
        if self.target_pos[0] < 5.0: repulsive_force[0] += 5.0 - self.target_pos[0]
        if self.target_pos[0] > self.cfg.MAP_SIZE - 5.0: repulsive_force[0] -= self.target_pos[0] - (
                    self.cfg.MAP_SIZE - 5.0)
        if self.target_pos[1] < 5.0: repulsive_force[1] += 5.0 - self.target_pos[1]
        if self.target_pos[1] > self.cfg.MAP_SIZE - 5.0: repulsive_force[1] -= self.target_pos[1] - (
                    self.cfg.MAP_SIZE - 5.0)

        if np.linalg.norm(repulsive_force) > 0.1:
            avoid_yaw = np.arctan2(repulsive_force[1], repulsive_force[0])
            self.target_yaw = 0.8 * self.target_yaw + 0.2 * avoid_yaw  # 融合避障方向

        self.target_pos[0] += self.target_v * np.cos(self.target_yaw) * self.cfg.DT
        self.target_pos[1] += self.target_v * np.sin(self.target_yaw) * self.cfg.DT
        self.target_pos = np.clip(self.target_pos, 0, self.cfg.MAP_SIZE)

    def _get_obs(self):
        """构建归一化的多机状态空间"""
        obs_all = []
        seen = [self._check_los(0)[0], self._check_los(1)[0]]

        for i in range(self.cfg.NUM_AGENTS):
            teammate_idx = 1 - i
            agent_obs = []

            # 1. 目标信息 (基于信息共享：哪怕自己没看见，队友看见了也能获取)
            is_seen_by_me = 1.0 if seen[i] else -1.0
            is_seen_by_team = 1.0 if seen[teammate_idx] else -1.0

            if seen[i] or seen[teammate_idx]:
                vec_t = self.target_pos - self.uav_pos[i]
                dist_t = np.linalg.norm(vec_t)
                angle_t = (np.arctan2(vec_t[1], vec_t[0]) - self.uav_yaw[i] + np.pi) % (2 * np.pi) - np.pi
                margin_d = (self.cfg.R_SENSE - dist_t) / self.cfg.R_SENSE
                margin_a = (self.cfg.FOV / 2 - np.abs(angle_t)) / (self.cfg.FOV / 2)
                agent_obs.extend([dist_t / self.cfg.MAP_SIZE, angle_t / np.pi, margin_d, margin_a])
            else:
                agent_obs.extend([0.0, 0.0, -1.0, -1.0])  # 丢失目标时的占位符

            agent_obs.extend([is_seen_by_me, is_seen_by_team])

            # 2. 队友信息
            vec_tm = self.uav_pos[teammate_idx] - self.uav_pos[i]
            dist_tm = np.linalg.norm(vec_tm)
            angle_tm = (np.arctan2(vec_tm[1], vec_tm[0]) - self.uav_yaw[i] + np.pi) % (2 * np.pi) - np.pi
            agent_obs.extend([dist_tm / self.cfg.MAP_SIZE, angle_tm / np.pi,
                              self.uav_v[teammate_idx] / self.cfg.MAX_V, self.uav_w[teammate_idx] / self.cfg.MAX_W])

            # 3. 自身状态
            agent_obs.extend([self.uav_v[i] / self.cfg.MAX_V, self.uav_w[i] / self.cfg.MAX_W])

            # 4. 障碍物信息 (最近的3个)
            obs_list = []
            for o in self.static_obs:
                d = self._get_boundary_dist(self.uav_pos[i], o)
                a = (np.arctan2(o['pos'][1] - self.uav_pos[i][1], o['pos'][0] - self.uav_pos[i][0]) - self.uav_yaw[
                    i] + np.pi) % (2 * np.pi) - np.pi
                obs_list.append([d / self.cfg.MAP_SIZE, a / np.pi, o['type']])

            obs_list.sort(key=lambda x: x[0])
            for j in range(3):
                if j < len(obs_list):
                    agent_obs.extend(obs_list[j])
                else:
                    agent_obs.extend([1.0, 0.0, -1])

            obs_all.append(agent_obs)

        return np.array(obs_all, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._generate_scenario()
        self.uav_v.fill(0.0)
        self.uav_w.fill(0.0)
        self.prev_actions.fill(0.0)
        self.step_count = 0
        self.lose_steps = 0
        return self._get_obs(), {}

    def step(self, actions):
        """核心交互逻辑：输入两架飞机的联合动作"""
        rewards = np.zeros(self.cfg.NUM_AGENTS)
        terminated = False
        info = {'reason': '', 'is_success': 0}

        # 1. 运动学更新
        for i in range(self.cfg.NUM_AGENTS):
            self.uav_v[i] = (actions[i][0] + 1.0) / 2.0 * self.cfg.MAX_V
            self.uav_w[i] = actions[i][1] * self.cfg.MAX_W
            self.uav_yaw[i] = (self.uav_yaw[i] + self.uav_w[i] * self.cfg.DT + np.pi) % (2 * np.pi) - np.pi
            self.uav_pos[i][0] += self.uav_v[i] * np.cos(self.uav_yaw[i]) * self.cfg.DT
            self.uav_pos[i][1] += self.uav_v[i] * np.sin(self.uav_yaw[i]) * self.cfg.DT

        self._update_target()
        self.step_count += 1

        # ==========================================
        # 2. 状态检测与各项奖励计算 (精细拆解版)
        # ==========================================
        seen_status = [self._check_los(0), self._check_los(1)]
        target_seen_by_any = seen_status[0][0] or seen_status[1][0]

        # 建立统计记录器
        stats = {'track': 0.0, 'standoff': 0.0, 'smooth': 0.0, 'danger': 0.0, 'dispersion': 0.0}
        r_lose_step_val = 0.0

        # 全局跟丢惩罚
        if not target_seen_by_any:
            self.lose_steps += 1
            r_lose_step_val = self.cfg.COEFF_LOSE_STEP
            rewards += r_lose_step_val
        else:
            self.lose_steps = 0

        for i in range(self.cfg.NUM_AGENTS):
            r_step_agent = 0.0
            my_seen, my_dist, my_angle = seen_status[i]

            # --- 跟踪与距离奖励 ---
            if my_seen:
                # 对齐奖励
                r_track_i = self.cfg.COEFF_TRACK_BASE * (1.0 - np.abs(my_angle) / (self.cfg.FOV / 2))
                # 最佳距离保持奖励
                ideal_dist = self.cfg.R_SENSE * 0.6
                dist_penalty = np.abs(my_dist - ideal_dist) / ideal_dist
                r_standoff_i = self.cfg.COEFF_STANDOFF * (1.0 - dist_penalty)

                r_step_agent += (r_track_i + r_standoff_i)
                stats['track'] += r_track_i
                stats['standoff'] += r_standoff_i

            # --- 动作平滑惩罚 ---
            act_diff = actions[i] - self.prev_actions[i]
            r_smooth_i = self.cfg.COEFF_SMOOTH * np.sum(act_diff ** 2)
            r_step_agent += r_smooth_i
            stats['smooth'] += r_smooth_i

            # --- 危险惩罚 (避障) ---
            min_d_obs = min([self._get_boundary_dist(self.uav_pos[i], o) for o in self.static_obs] + [float('inf')])
            if min_d_obs < 1.5:
                r_danger_i = self.cfg.COEFF_DANGER * (1.5 - min_d_obs)
                r_step_agent += r_danger_i
                stats['danger'] += r_danger_i

            # --- 协同包抄奖励 ---
            if seen_status[0][0] and seen_status[1][0]:
                vec1 = self.uav_pos[0] - self.target_pos
                vec2 = self.uav_pos[1] - self.target_pos
                cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-5)
                r_disp_i = self.cfg.COEFF_DISPERSION * (1.0 - np.abs(cos_theta))
                r_step_agent += r_disp_i
                stats['dispersion'] += r_disp_i

            # 裁剪单步奖励并汇总
            rewards[i] += np.clip(r_step_agent, -2.0, 2.0)

        self.prev_actions = np.copy(actions)

        # ==========================================
        # 3. 终止条件判定与事件奖励
        # ==========================================
        # 用于记录一次性触发的事件奖励
        event_rewards = {'collision': 0.0, 'lose_done': 0.0, 'success': 0.0}

        # A. 碰撞检测
        for i in range(self.cfg.NUM_AGENTS):
            if self.uav_pos[i][0] < 0 or self.uav_pos[i][0] > self.cfg.MAP_SIZE or \
                    self.uav_pos[i][1] < 0 or self.uav_pos[i][1] > self.cfg.MAP_SIZE:
                rewards[i] += self.cfg.COEFF_COLLISION
                event_rewards['collision'] = self.cfg.COEFF_COLLISION
                terminated = True;
                info['reason'] = 'out_of_bounds'
            for o in self.static_obs:
                if self._get_boundary_dist(self.uav_pos[i], o) < 0:
                    rewards[i] += self.cfg.COEFF_COLLISION
                    event_rewards['collision'] = self.cfg.COEFF_COLLISION
                    terminated = True;
                    info['reason'] = 'collision_obs'

        # B. 飞机互撞
        if np.linalg.norm(self.uav_pos[0] - self.uav_pos[1]) < self.cfg.SAFE_DIST_UAV:
            rewards += self.cfg.COEFF_COLLISION
            event_rewards['collision'] = self.cfg.COEFF_COLLISION
            terminated = True;
            info['reason'] = 'collision_uavs'

        # C. 任务失败判定 (彻底丢失)
        if self.lose_steps >= self.cfg.TARGET_LOSE_LIMIT:
            rewards += self.cfg.COEFF_LOSE_DONE
            event_rewards['lose_done'] = self.cfg.COEFF_LOSE_DONE
            terminated = True;
            info['reason'] = 'target_lost'

        # D. 截断/胜利判定
        truncated = self.step_count >= self.cfg.MAX_STEPS
        if truncated and not terminated:
            rewards += self.cfg.COEFF_SUCCESS
            event_rewards['success'] = self.cfg.COEFF_SUCCESS
            info['is_success'] = 1
            info['reason'] = 'max_steps_survived'

        # ==========================================
        # 4. 装填 Info 字典供数据统计
        # ==========================================
        info['reward_comps'] = {
            # 针对双机取平均值，更能反映全局策略的倾向
            'r_track_mean': stats['track'] / self.cfg.NUM_AGENTS,
            'r_standoff_mean': stats['standoff'] / self.cfg.NUM_AGENTS,
            'r_smooth_mean': stats['smooth'] / self.cfg.NUM_AGENTS,
            'r_danger_mean': stats['danger'] / self.cfg.NUM_AGENTS,
            'r_dispersion_mean': stats['dispersion'] / self.cfg.NUM_AGENTS,
            # 全局与事件状态
            'r_lose_step': r_lose_step_val,
            'r_collision_event': event_rewards['collision'],
            'r_lose_done_event': event_rewards['lose_done'],
            'r_success_event': event_rewards['success']
        }

        return self._get_obs(), rewards, terminated, truncated, info