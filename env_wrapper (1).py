import jax
import jax.numpy as jnp

from env import MJXLeapHandEnv
from observation_buffer import (
    build_asymmetric_observation,
    ACTOR_OBS_DIM,
    CRITIC_OBS_DIM,
    _extract_tactile_features,   # NOUVEAU — on importe la fonction de calcul tactile
)
from tactile import TactileSystem, TACTILE_DIM
from recurrl_jax.utils.wrappers import MJXGymWrapper
from recurrl_jax.utils.quat_utils import rotate_vec_by_quat


class LeapHandGymWrapper(MJXGymWrapper):
    def __init__(
        self,
        num_envs: int = 1,
        use_domain_randomization: bool = False,
        action_scale: float = 0.6,
        action_ema_alpha: float = 0.0,
        grasp_cache_path: str = None,
        wrench_force_scale: float = 5.0,
        wrench_torque_scale: float = 0.5,
        alive_bonus: float = 2.0,
        wrench_resistance_scale: float = 1.0,
        slip_vel_scale: float = 0.1,
        torque_scale: float = 0.0001,
        action_rate_scale: float = 0.01,
        wrench_ramp_alpha: float = 0.8,
        wrench_push_steps: tuple = (20, 80),
        wrench_rest_steps: tuple = (10, 50),
        # NOUVEAU — guidage
        fingertip_proximity_scale: float = 5.0,
        fingertip_contact_scale: float = 2.0,
        self_collision_scale: float = 1.0,
        # NOUVEAU — coefficients de recompense tactile
        contact_bonus_scale: float = 1.0,
        pressure_bonus_scale: float = 0.5,
        palm_bonus_scale: float = 0.3,
        low_pressure_scale: float = 0.5,
        high_pressure_scale: float = 0.2,
        **kwargs,
    ):
        # stockage des parametres existants (identique a Eduardo)
        self._use_dr = use_domain_randomization
        self._action_scale = action_scale
        self._action_ema_alpha = action_ema_alpha
        self._grasp_cache_path = grasp_cache_path
        self._wrench_force_scale = wrench_force_scale
        self._wrench_torque_scale = wrench_torque_scale
        self._alive_bonus = alive_bonus
        self._wrench_resistance_scale = wrench_resistance_scale
        self._slip_vel_scale = slip_vel_scale
        self._torque_scale = torque_scale
        self._action_rate_scale = action_rate_scale
        self._wrench_ramp_alpha = wrench_ramp_alpha
        self._wrench_push_steps = tuple(wrench_push_steps)
        self._wrench_rest_steps = tuple(wrench_rest_steps)

        # NOUVEAU — stockage des coefficients
        self._fingertip_proximity_scale = fingertip_proximity_scale
        self._fingertip_contact_scale = fingertip_contact_scale
        self._self_collision_scale = self_collision_scale
        self._contact_bonus_scale = contact_bonus_scale
        self._pressure_bonus_scale = pressure_bonus_scale
        self._palm_bonus_scale = palm_bonus_scale
        self._low_pressure_scale = low_pressure_scale
        self._high_pressure_scale = high_pressure_scale

        super().__init__(
            obs_dim=CRITIC_OBS_DIM,        # 100
            action_dim=16,
            num_envs=num_envs,
            policy_obs_dim=ACTOR_OBS_DIM,  # 58
            **kwargs,
        )

        # initialisation du systeme tactile (identique a Eduardo)
        self.key, tac_key = jax.random.split(self.key)
        dt = self.env.mj_model.opt.timestep * self.env.control_freq_inv
        self.tactile_system = TactileSystem(
            mj_model=self.env.mj_model,
            num_envs=num_envs,
            key=tac_key,
            dt=float(dt),
        )
        self._tactile_obs = jnp.zeros((num_envs, TACTILE_DIM))

        # NOUVEAU — initialisation des features tactiles extraites
        # (utilisees a la fois pour l obs et pour la recompense)
        self._contact_fingers  = jnp.zeros((num_envs, 4))
        self._pressure_fingers = jnp.zeros((num_envs, 4))
        self._contact_palm     = jnp.zeros((num_envs, 1))
        self._pressure_palm    = jnp.zeros((num_envs, 1))

        # NOUVEAU — passer les coefficients tactiles a env.py via _make_env
        # (deja fait dans _make_env ci-dessous)

    def _make_env(self, key):
        return MJXLeapHandEnv(
            xml_path='xmls/scene_mjx.xml',
            num_envs=self.num_envs,
            key=key,
            action_scale=self._action_scale,
            action_ema_alpha=self._action_ema_alpha,
            use_domain_randomization=self._use_dr,
            grasp_cache_path=self._grasp_cache_path,
            wrench_force_scale=self._wrench_force_scale,
            wrench_torque_scale=self._wrench_torque_scale,
            alive_bonus=self._alive_bonus,
            wrench_resistance_scale=self._wrench_resistance_scale,
            slip_vel_scale=self._slip_vel_scale,
            torque_scale=self._torque_scale,
            action_rate_scale=self._action_rate_scale,
            wrench_ramp_alpha=self._wrench_ramp_alpha,
            wrench_push_steps=self._wrench_push_steps,
            wrench_rest_steps=self._wrench_rest_steps,
            # NOUVEAU — coefficients tactiles passes a env.py
            fingertip_proximity_scale=self._fingertip_proximity_scale,
            fingertip_contact_scale=self._fingertip_contact_scale,
            self_collision_scale=self._self_collision_scale,
            contact_bonus_scale=self._contact_bonus_scale,
            pressure_bonus_scale=self._pressure_bonus_scale,
            palm_bonus_scale=self._palm_bonus_scale,
            low_pressure_scale=self._low_pressure_scale,
            high_pressure_scale=self._high_pressure_scale,
        )

    def reset(self, env_ids=None, **kwargs):
        if env_ids is None:
            reset_mask = jnp.ones(self.num_envs, dtype=bool)
        else:
            reset_mask = jnp.zeros(self.num_envs, dtype=bool).at[jnp.array(env_ids)].set(True)

        # reset tactile (identique a Eduardo)
        self.tactile_system.reset_states(reset_mask)
        self._tactile_obs = jnp.where(
            reset_mask[:, None],
            jnp.zeros((self.num_envs, TACTILE_DIM)),
            self._tactile_obs
        )

        # NOUVEAU — reset des features tactiles extraites
        self._contact_fingers  = jnp.where(reset_mask[:, None], jnp.zeros((self.num_envs, 4)), self._contact_fingers)
        self._pressure_fingers = jnp.where(reset_mask[:, None], jnp.zeros((self.num_envs, 4)), self._pressure_fingers)
        self._contact_palm     = jnp.where(reset_mask[:, None], jnp.zeros((self.num_envs, 1)), self._contact_palm)
        self._pressure_palm    = jnp.where(reset_mask[:, None], jnp.zeros((self.num_envs, 1)), self._pressure_palm)

        return super().reset(**kwargs)

    def step(self, actions):
        actions_jax = jnp.asarray(actions)

        # NOUVEAU — extraire les features tactiles du step precedent UNE SEULE FOIS
        # Elles seront utilisees a la fois pour la recompense (via env.step)
        # et pour l observation (via _get_obs)
        (
            self._contact_fingers,
            self._pressure_fingers,
            self._contact_palm,
            self._pressure_palm,
        ) = _extract_tactile_features(self._tactile_obs)

        # NOUVEAU — stocker dans env pour que _leap_reward_fn y accede via extra_state_fn
        self.env.current_contact_fingers  = self._contact_fingers
        self.env.current_pressure_fingers = self._pressure_fingers
        self.env.current_contact_palm     = self._contact_palm
        self.env.current_pressure_palm    = self._pressure_palm

        # avancer la simulation (identique a Eduardo)
        _, reward, reset_mask, termination, info, *_ = self.env.step(actions_jax)

        self.last_actions = jnp.where(
            reset_mask[:, None],
            jnp.zeros((self.num_envs, self.action_dim)),
            actions_jax
        )

        # mettre a jour le tactile avec les nouvelles donnees physiques (identique a Eduardo)
        self.key, tac_key = jax.random.split(self.key)
        self._tactile_obs = self.tactile_system.step(
            self.env.mjx_data_batch, tac_key, reset_mask
        )

        obs = self._normalize(self._get_obs())
        reward = jnp.nan_to_num(reward * self.reward_scale, nan=0.0, posinf=0.0, neginf=0.0)
        truncation = jnp.logical_and(reset_mask, jnp.logical_not(termination))
        return obs, reward, termination, truncation, info

    def _get_obs(self) -> jnp.ndarray:
        mjx_data = self.env.mjx_data_batch
        qpos = mjx_data.qpos
        qvel = mjx_data.qvel
        obj_quat = qpos[:, 19:23]

        # action precedente (stockee par Eduardo dans self.prev_actions)
        prev_action = (
            self.env.prev_actions
            if self.env.prev_actions is not None
            else jnp.zeros((self.num_envs, 16))
        )

        # couples moteurs (critic)
        motor_torques = mjx_data.qfrc_actuator[:, :16]

        # positions fingertips (critic)
        ft_pos = self.env.get_fingertip_positions(mjx_data)
        fingertip_pos = ft_pos.reshape(self.num_envs, 12)

        # position paume (critic)
        palm_pos = self.env.get_palm_position(mjx_data)

        # chute detectee (critic)
        cube_z = qpos[:, 18]
        fall_flag = (cube_z < self.env.reset_height_threshold).astype(jnp.float32)

        return build_asymmetric_observation(
            joint_angles  = qpos[:, :16],
            prev_action   = prev_action,
            joint_vel     = qvel[:, :16],
            # NOUVEAU — on passe les features deja extraites, pas le vecteur brut
            contact_fingers  = self._contact_fingers,
            pressure_fingers = self._pressure_fingers,
            contact_palm     = self._contact_palm,
            pressure_palm    = self._pressure_palm,
            motor_torques = motor_torques,
            fingertip_pos = fingertip_pos,
            obj_pos       = qpos[:, 16:19],
            obj_quat      = obj_quat,
            obj_linvel    = qvel[:, 16:19],
            obj_angvel    = rotate_vec_by_quat(qvel[:, 19:22], obj_quat),
            fall_flag     = fall_flag,
            palm_pos      = palm_pos,
        )
