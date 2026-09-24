"""lift_cup 任务的环境装配：复用 hang_cup 场景，换原生 Lift-Cube MDP。

2026-09-21 用户裁决新建：与 HangCupEnvCfg 并列的独立任务
（Saw-LiftCup-FrankaDual-v0），只做"右手抓起杯子举离桌面 15cm"——
Isaac-Lift-Cube-Franka 配方在同场景的最小复刻。场景/机器人/相机/
接触传感器全部复用 env_cfg.HangCupSceneCfg（传感器挂着不用零成本），
MDP 函数见 mdp_lift.py。观测去掉 z_latch（无状态机）。

2026-09-23 用户裁决向官方配方对齐（血泪补丁表现不如官方）：
  ① reach 改用官方同款 ee_frame（FrameTransformer，干活左臂
    panda_hand + 0.1034 指尖偏移）对杯心测距；杯 root 即几何/碰撞
    中心（同官方 cube root 定义），指尖中心与抓取位重合，奖励峰可达；
  ② lift 还原为官方阶跃（杯心高过静置 5cm 给钱，w=15）；
  ③ action_rate/joint_vel 对齐官方：−1e-4 起步 + 课程 10000 步升
    −1e-1（joint_vel 罚 robot_right——右臂干活）；
  ④ 并行默认 1024（simulation.yaml rl_num_envs）；
  ⑤ PPO learning_rate 1e-4（agents/rsl_rl_ppo_cfg.py）。
保留：成功奖 +10（带速度门）、掉杯终止（纯收口不罚款）、全部
数值保险丝。动作契约改为 8 维左臂（右手不参与，2026-09-23）。
掉杯罚款与臂碰桌/臂体碰杯惩罚已按 2026-09-23 裁决关闭（策略
学会远离杯子回避惩罚）。
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from simulation.robots import get_spec
from . import mdp, mdp_lift
from .env_cfg import (HangCupSceneCfg, load_sim_config, camera_names,
                      look_at_quat_opengl)


# --------------------------------------------------------------------------- #
# MDP 配置
# --------------------------------------------------------------------------- #

@configclass
class ActionsCfg:
    # 臂 delta + 夹爪 binary（2026-09-21 用户裁决："夹爪对神经网络的
    # 表现和官方一样"——Isaac-Lift 官方 BinaryJointPositionActionCfg
    # 语义：正号全开/负号全闭、无积分无记忆，见 mdp.py 类注释）。
    # 8 维左臂动作（2026-09-23 用户裁决"右手不参与"）：
    # LeftArmDeltaBinaryGripperAction = [左臂7 delta, 左爪 binary]，
    # 右臂每步显式锁 home + 全开（mdp.py 类注释）。网络输出 16→8，
    # last_action 观测 16→8（总 55→47），旧 ckpt 不兼容需重训；
    # （更早的右臂冻结/网络锁 0 两版方案已同步退役，右臂不再参与。）
    arm = mdp.LeftArmDeltaBinaryGripperActionCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        proprio = ObsTerm(func=mdp.proprio_pos)
        joint_vel = ObsTerm(func=mdp.proprio_vel)
        cup = ObsTerm(func=mdp.cup_pose)
        # last_action 钳 ±2（2026-09-21 两次 iter ~400/1282 崩溃确诊）：
        # 环境动作 clamp [-1,1] 后任务奖励对 |mu|>1 不敏感，mu 随机游走
        # 无界漂移（实测 raw action ~1e4），未钳的 last_action 把它灌进
        # critic 输入 → value loss 平方爆 inf。有意义范围就是 [-1,1]，
        # ±2 留足余量。这是观测里唯一的非物理量，物理保险丝管不到它。
        last_action = ObsTerm(func=base_mdp.last_action, clip=(-2.0, 2.0))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CameraObsCfg(ObsGroup):
        """视觉观测组（与 hang_cup 同构）：训练剥离，录制/像素策略用。"""
        rgb = ObsTerm(func=mdp.camera_rgb, params={"name": "camera_front"})
        depth = ObsTerm(func=mdp.camera_depth, params={"name": "camera_front"})

        def __post_init__(self):
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    camera: CameraObsCfg = CameraObsCfg()


@configclass
class RewardsCfg:
    # 官方 Lift-Cube 配方对齐版（2026-09-23 用户裁决③）：
    #   reach（1−tanh(d/0.1)，w=1.0，ee_frame 测距，见 mdp_lift.rew_reach）
    #   lift（阶跃：杯心高过静置 5cm 给钱，w=15.0，同官方 lifting_object）
    #   action_rate（−1e-4 起步，课程 10000 步升 −1e-1，同官方）
    #   joint_vel（−1e-4 起步，同上课程；官方罚整机，本任务左臂干活
    #   只罚 robot_left——右臂已冻结，罚它恒 ≈0 形同虚设）
    # 保留：成功奖 +10（带速度门）。掉杯只终止不罚款、无任何
    # 接触惩罚（2026-09-23 裁决关闭，与官方配方一致）。
    # 注意：本类不能有 RewardTermCfg 以外的类属性（manager 全遍历）。
    reach = RewTerm(func=mdp_lift.rew_reach, weight=1.0)
    lift = RewTerm(func=mdp_lift.rew_lift, weight=45.0)
    lift2 = RewTerm(func=mdp_lift.rew_lift2, weight=45.0)
    success_bonus = RewTerm(func=mdp_lift.rew_success, weight=45.0)
    action_rate = RewTerm(func=mdp.rew_action_rate, weight=-1e-4)
    joint_vel = RewTerm(func=base_mdp.joint_vel_l2, weight=-1e-4,
                        params={"asset_cfg": SceneEntityCfg("robot_left")})
    # 掉杯罚款/臂碰桌/臂体碰杯惩罚已关闭（2026-09-23 用户裁决）：
    # 实测策略初期探索碰掉杯几次后学会"远离杯子"回避惩罚，与任务
    # 目标直接相悖；官方 Lift-Cube 配方也无任何接触惩罚。掉杯仍由
    # TerminationsCfg.dropped 终止回合（只收口不塑形）。


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp_lift.lift_success)
    dropped = DoneTerm(func=mdp.cup_dropped)


@configclass
class CurriculumCfg:
    """课程（对齐官方 LiftEnvCfg.CurriculumCfg，2026-09-23 用户裁决③）：
    前 10000 步把动作率/关节速度惩罚从 −1e-4 升到 −1e-1——先学会
    抓举，再修剪抖动与多余动作，避免重惩罚压制早期探索。"""
    action_rate = CurrTerm(
        func=base_mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000})
    joint_vel = CurrTerm(
        func=base_mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000})


@configclass
class EventsCfg:
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    # 杯摩擦材质（μ_s=1.2/μ_d=1.0，2026-09-18 裁决理由见 env_cfg 同义项）
    set_cup_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("cup"),
                "static_friction_range": (1.2, 1.2),
                "dynamic_friction_range": (1.0, 1.0),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 1})
    randomize_cup = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("cup"),
                "pose_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.1)},
                "velocity_range": {}})


# --------------------------------------------------------------------------- #
# EnvCfg
# --------------------------------------------------------------------------- #

@configclass
class LiftCupEnvCfg(ManagerBasedRLEnvCfg):
    scene: HangCupSceneCfg = HangCupSceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        cfg = load_sim_config()
        env_cfg = cfg["env"]

        self.decimation = env_cfg["decimation"]
        # 20s @15Hz = 300 步（2026-09-23 用户裁决：10s/150 步太短，
        # 策略学会夹持后来不及抬升回合就结束了——官方 5s@50Hz=250
        # 步，这里决策步数给到 1.2 倍。杯落地由 TerminationsCfg.dropped
        # 提前收口，不落地就充分利用回合长度）。
        self.episode_length_s = 20.0
        self.sim.dt = env_cfg["sim_dt"]
        self.sim.render_interval = self.decimation
        self.scene.num_envs = env_cfg["num_envs"]

        # 穿模彻查裁决（同 hang_cup，理由见 env_cfg.__post_init__）
        self.sim.physx.min_position_iteration_count = 8
        self.sim.physx.min_velocity_iteration_count = 4

        arms = get_spec(cfg["robot"]).build()
        self.scene.robot_left = arms["left"].replace(
            prim_path="{ENV_REGEX_NS}/RobotLeft")
        self.scene.robot_right = arms["right"].replace(
            prim_path="{ENV_REGEX_NS}/RobotRight")

        # ee_frame（2026-09-23 用户裁决①，官方同款）：干活左臂
        # panda_hand + 0.1034 指尖偏移，reach 测距用——同官方
        # object_ee_distance 的 FrameTransformer 配置（单 target，
        # target_pos_w 形状 (N,1,3)）。杯 root 即几何中心，指尖中心
        # 与抓取位重合，奖励峰可达。debug_vis 保持关（训练无头）。
        # 只挂干活的左臂：右臂已退出任务（LeftArm 动作项每步锁
        # home），若保留双手 target + min 取距，锁死手的距离是
        # 常量——一旦它比左臂近，min 对左臂动作的梯度恒 0。
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/RobotLeft/panda_link0",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/RobotLeft/panda_hand",
                    name="ee_left",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )

        # 杯心 debug 可视化（2026-09-23）：FrameTransformer 以杯自身为
        # source/target，debug_vis=True 时在杯 root（=几何/碰撞中心，
        # rew_reach 的目标点）画 RGB 坐标轴。默认关——训练无头无渲染；
        # train_lift.py --debug-cup 打开（改 cfg 上的 debug_vis 即可，
        # 场景在 env 创建时才实例化）。同理把 ee_frame.debug_vis 置
        # True 可同时看双手指尖帧。
        self.scene.cup_center_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Cup",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Cup",
                    name="cup_center",
                ),
            ],
        )

        for name, cam in cfg["cameras"].items():
            setattr(self.scene, f"camera_{name}", CameraCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Camera_{name}",
                update_period=0.0,
                update_latest_camera_pose=True,
                width=cam["width"], height=cam["height"],
                data_types=cam["data_types"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=cam["focal_length_mm"],
                    clipping_range=(0.05, 10.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=tuple(cam["pos"]),
                    rot=look_at_quat_opengl(cam["pos"], cam["look_at"]),
                    convention="opengl",
                ),
            ))


def make_rl_cfg(num_envs: int | None = None) -> LiftCupEnvCfg:
    """RL 训练用 env cfg：剥离全部相机（同 env_cfg.make_rl_cfg 的理由）。

    注意 ee_frame 不是相机，reach 奖励依赖它，不能剥离。"""
    cfg = LiftCupEnvCfg()
    cfg.scene.num_envs = int(num_envs or load_sim_config()["env"]["rl_num_envs"])
    for name in camera_names():
        delattr(cfg.scene, f"camera_{name}")
    cfg.observations.camera = None
    return cfg
