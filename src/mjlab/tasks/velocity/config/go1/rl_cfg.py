from dataclasses import dataclass, field

from mjlab.rl import (
  RslRlDistillationAlgorithmCfg,
  RslRlDistillationRunnerCfg,
  RslRlDistillationStudentTeacherCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoActorCriticCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class UnitreeGo1PPORunnerCfg(RslRlOnPolicyRunnerCfg):
  policy: RslRlPpoActorCriticCfg = field(
    default_factory=lambda: RslRlPpoActorCriticCfg(
      init_noise_std=1.0,
      actor_obs_normalization=False,
      critic_obs_normalization=False,
      actor_hidden_dims=(512, 256, 128),
      critic_hidden_dims=(512, 256, 128),
      activation="elu",
    )
  )
  algorithm: RslRlPpoAlgorithmCfg = field(
    default_factory=lambda: RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    )
  )
  experiment_name: str = "go1_velocity"
  save_interval: int = 50
  num_steps_per_env: int = 24
  max_iterations: int = 10_000


@dataclass
class UnitreeGo1DistillRunnerCfg(RslRlDistillationRunnerCfg):
  num_steps_per_env: int = 120
  max_iterations: int = 10_000
  save_interval: int = 500
  experiment_name: str = "go1_velocity"
  obs_groups: dict[str, list[str]] = field(
    default_factory=lambda: {"policy": ["policy"], "teacher": ["policy"]},
  )
  policy: RslRlDistillationStudentTeacherCfg = field(
    default_factory=lambda: RslRlDistillationStudentTeacherCfg(
      init_noise_std=0.1,
      noise_std_type="scalar",
      student_obs_normalization=False,
      teacher_obs_normalization=False,
      student_hidden_dims=(128, 128, 128),
      teacher_hidden_dims=(512, 256, 128),
      activation="elu",
    )
  )
  algorithm: RslRlDistillationAlgorithmCfg = field(
    default_factory=lambda: RslRlDistillationAlgorithmCfg(
      num_learning_epochs=2,
      learning_rate=1.0e-3,
      gradient_length=15,
    )
  )
