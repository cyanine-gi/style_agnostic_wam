"""rsl_rl PPO 配置（RL 库细节只进不出——外层一律经 rl/facade.py 访问）。"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (RslRlOnPolicyRunnerCfg,
                                RslRlPpoActorCriticCfg,
                                RslRlPpoAlgorithmCfg)


@configclass
class HangCupPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "hang_cup_franka_dual"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
