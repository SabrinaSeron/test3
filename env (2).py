from etils import epath

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import mujoco
from mujoco import mjx

from functools import partial
from rewards import compute_catching_reward, check_termination
from domain_randomization_model import build_leap_hand_dr_params
from recurrl_jax.utils.domain_randomization import create_batched_randomized_models
from recurrl_jax.utils.mjx_env import EnvSpec, mjx_training_step, _mjx_reset_jit


_LEAP_DEFAULT_POSE = (0.8, 0.0, 0.8, 0.8, 0.8, 0.0, 0.8, 0.8, 0.8, 0.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.0)

# Pose ouverte pour commencer avec le cube sur la paume
# Les doigts sont légèrement courbés mais pas fermés
_OPEN_POSE = jnp.array([
    0.3, 0.0, 0.1, 0.1,   # index : légèrement ouvert
    0.3, 0.0, 0.1, 0.1,   # majeur
    0.3, 0.0, 0.1, 0.1,   # annulaire
    0.5, 0.3, 0.2, 0.1,   # pouce : légèrement plié
])


def build_leap_spec(mj_model) -> EnvSpec:
    fingertip_geom_ids = tuple(
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("th_tip", "if_tip", "mf_tip", "rf_tip")
    )
    cube_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "cube")
    return EnvSpec(
        n_dofs=16,
        dof_slice=(0, 16),
        obj_pos_slice=(16, 19),
        obj_quat_slice=(19, 23),
        obj_linvel_slice=(16, 19),
        obj_angvel_slice=(19, 22),
        default_pose=_LEAP_DEFAULT_POSE,
        contact_geom_ids=fingertip_geom_ids,
        obj_geom_id=cube_geom_id,
    )


def _leap_reward_fn(state, reset_height_threshold,
                    slip_vel_scale=0.1, torque_scale=0.0001,
                    alive_bonus=2.0, drop_penalty=-1000.0,
                    fingertip_proximity_scale=5.0,
                    fingertip_contact_scale=2.0,
                    self_collision_scale=1.0,
                    contact_bonus_scale=1.0,
                    pressure_bonus_scale=0.5,
                    palm_bonus_scale=0.3,
                    low_pressure_scale=0.5,
                    high_pressure_scale=0.2):
    reward, info = compute_catching_reward(
        obj_linvel         = state['obj_linvel'],
        obj_angvel         = state['obj_angvel'],
        obj_pos            = state['obj_pos'],
        torques            = state['torques'],
        contact_fingers    = state['contact_fingers'],
        pressure_fingers   = state['pressure_fingers'],
        contact_palm       = state['contact_palm'],
        pressure_palm      = state['pressure_palm'],
        fingertip_pos      = state['fingertip_pos'],
        fingertip_contact  = state['contact'],          # depuis mjx_env.py
        self_collision     = state['self_collision'],   # NOUVEAU
        reset_height_threshold    = reset_height_threshold,
        slip_vel_scale            = slip_vel_scale,
        torque_scale              = torque_scale,
        alive_bonus               = alive_bonus,
        drop_penalty              = drop_penalty,
        fingertip_proximity_scale = fingertip_proximity_scale,
        fingertip_contact_scale   = fingertip_contact_scale,
        self_collision_scale      = self_collision_scale,
        contact_bonus_scale       = contact_bonus_scale,
        pressure_bonus_scale      = pressure_bonus_scale,
        palm_bonus_scale          = palm_bonus_scale,
        low_pressure_scale        = low_pressure_scale,
        high_pressure_scale       = high_pressure_scale,
    )
    new_prev_aux = jnp.zeros(state['obj_pos'].shape[0])
    return reward, info, new_prev_aux


def _leap_termination_fn(state, progress_buf, max_episode_length, reset_height_threshold):
    return check_termination(
        object_pos=state['obj_pos'],
        progress_buf=progress_buf,
        max_episode_length=max_episode_length,
        reset_height_threshold=reset_height_threshold,
    )


def _leap_reset_state_fn(key, data_batch, reset_mask, spec, use_dr, base_mjx, grasp_cache, grasp_cache_size):
    num_envs = reset_mask.shape[0]

    keys = jr.split(key, num_envs + 1)
    env_keys = keys[:-1]
    keys_split = jax.vmap(lambda k: jr.split(k, 3))(env_keys)
    hand_keys     = keys_split[:, 0]
    cube_pos_keys = keys_split[:, 1]
    cube_quat_keys = keys_split[:, 2]

    joint_lower = base_mjx.jnt_range[:16, 0]
    joint_upper = base_mjx.jnt_range[:16, 1]

    # Pose ouverte avec petit bruit
    hand_noise = jax.vmap(
        lambda k: jr.uniform(k, shape=(16,), minval=-0.05, maxval=0.05)
    )(hand_keys)
    open_dofs = jnp.clip(_OPEN_POSE + hand_noise, joint_lower, joint_upper)

    # Cube posé sur la paume, légèrement au-dessus
    cube_positions = jax.vmap(lambda k: jr.uniform(
        k, shape=(3,),
        minval=jnp.array([0.08, -0.01, 0.04]),
        maxval=jnp.array([0.12,  0.01, 0.06]),
    ))(cube_pos_keys)

    # Orientation légèrement aléatoire du cube (axe Z seulement)
    def random_quat_z(k):
        angle = jr.uniform(k, minval=-0.3, maxval=0.3)
        return jnp.array([
            jnp.cos(angle / 2), 0.0, 0.0, jnp.sin(angle / 2)
        ])
    cube_quats = jax.vmap(random_quat_z)(cube_quat_keys)

    randomized_qpos = jnp.concatenate([open_dofs, cube_positions, cube_quats], axis=1)
    new_qpos = jnp.where(reset_mask[:, None], randomized_qpos, data_batch.qpos)
    new_ctrl = new_qpos[:, :16]
    new_qvel = jnp.zeros_like(data_batch.qvel)
    return data_batch.replace(qpos=new_qpos, qvel=new_qvel, ctrl=new_ctrl)


class MJXLeapHandEnv:
    def __init__(self, xml_path: str, num_envs: int, key: jax.Array,
                 action_scale: float = 0.6, use_domain_randomization: bool = False,
                 grasp_cache_path: str = None, action_ema_alpha: float = 0.0,
                 wrench_force_scale: float = 5.0, wrench_torque_scale: float = 0.5,
                 slip_vel_scale: float = 0.1, torque_scale: float = 0.0001,
                 alive_bonus: float = 2.0,
                 wrench_resistance_scale: float = 1.0, action_rate_scale: float = 0.01,
                 wrench_ramp_alpha: float = 0.8,
                 wrench_push_steps: tuple = (20, 80),
                 wrench_rest_steps: tuple = (10, 50),
                 fingertip_proximity_scale: float = 5.0,
                 fingertip_contact_scale: float = 2.0,
                 self_collision_scale: float = 1.0,
                 contact_bonus_scale: float = 1.0,
                 pressure_bonus_scale: float = 0.5,
                 palm_bonus_scale: float = 0.3,
                 low_pressure_scale: float = 0.5,
                 high_pressure_scale: float = 0.2):

        self.mjx_path = epath.Path(xml_path).as_posix()
        self.num_envs = num_envs
        self.key = key
        self.action_scale = action_scale
        self.action_ema_alpha = action_ema_alpha
        self.use_domain_randomization = use_domain_randomization

        self.progress_buf = jnp.zeros(num_envs, dtype=jnp.int32)
        self.initial_dof_pos = None
        self.prev_smoothed_actions = None
        self.reset_height_threshold = -0.05
        self.max_episode_length = 500
        self.control_freq_inv = 5
        self.angvel_z_smooth = jnp.zeros(num_envs)

        self.wrench_force_scale = wrench_force_scale
        self.wrench_torque_scale = wrench_torque_scale
        self.wrench_ramp_alpha = wrench_ramp_alpha
        self.wrench_push_steps = wrench_push_steps
        self.wrench_rest_steps = wrench_rest_steps
        self.current_wrench = jnp.zeros((num_envs, 6))
        self.wrench_target = jnp.zeros((num_envs, 6))
        self.wrench_pushing = jnp.zeros(num_envs, dtype=bool)
        self.key, _wk = jr.split(self.key)
        self.wrench_countdown = jr.randint(
            _wk, (num_envs,), wrench_rest_steps[0], wrench_rest_steps[1] + 1
        )
        self.wrench_resistance_scale = wrench_resistance_scale
        self.action_rate_scale = action_rate_scale
        self.prev_actions = None

        # features tactiles
        self.current_contact_fingers  = jnp.zeros((num_envs, 4))
        self.current_pressure_fingers = jnp.zeros((num_envs, 4))
        self.current_contact_palm     = jnp.zeros((num_envs, 1))
        self.current_pressure_palm    = jnp.zeros((num_envs, 1))
        self.current_fingertip_pos    = jnp.zeros((num_envs, 12))

        self._reward_fn = partial(
            _leap_reward_fn,
            slip_vel_scale=slip_vel_scale,
            torque_scale=torque_scale,
            alive_bonus=alive_bonus,
            drop_penalty=-(alive_bonus * self.max_episode_length),
            fingertip_proximity_scale=fingertip_proximity_scale,
            fingertip_contact_scale=fingertip_contact_scale,
            self_collision_scale=self_collision_scale,
            contact_bonus_scale=contact_bonus_scale,
            pressure_bonus_scale=pressure_bonus_scale,
            palm_bonus_scale=palm_bonus_scale,
            low_pressure_scale=low_pressure_scale,
            high_pressure_scale=high_pressure_scale,
        )

        # grasp cache (gardé pour compatibilité mais non utilisé en mode open-hand)
        self.grasp_cache = None
        self.grasp_cache_size = 0
        if grasp_cache_path is not None:
            cache_data = np.load(grasp_cache_path)
            self.grasp_cache = jnp.array(cache_data)
            self.grasp_cache_size = self.grasp_cache.shape[0]
            print(f"Loaded grasp cache from {grasp_cache_path}: {self.grasp_cache_size} grasps")

        self.mj_model = mujoco.MjModel.from_xml_path(self.mjx_path)
        self.mj_data  = mujoco.MjData(self.mj_model)
        self.mjx_model = mjx.put_model(self.mj_model)

        self.obj_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.if_tip_id   = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "if_tip")
        self.mf_tip_id   = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "mf_tip")
        self.rf_tip_id   = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "rf_tip")
        self.th_tip_id   = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "th_tip")
        self.palm_id     = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "palm")

        # IDs des geoms fingertips pour détecter la self-collision
        self._fingertip_geom_ids = jnp.array([
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("th_tip", "if_tip", "mf_tip", "rf_tip")
        ])
        self._cube_geom_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "cube"
        )

        self.base_mjx  = self.mjx_model
        self.dr_params = build_leap_hand_dr_params(self.mj_model)
        self.spec      = build_leap_spec(self.mj_model)

        self.mjx_model_batch, self.mjx_data_batch = self._create_batch()

        self.num_dofs         = self.mj_model.nu
        self.ctrl_range       = self.mjx_model.actuator_ctrlrange
        self.joint_lower_limits = self.mjx_model.jnt_range[:, 0][:16]
        self.joint_upper_limits = self.mjx_model.jnt_range[:, 1][:16]

    def _create_batch(self):
        if self.use_domain_randomization:
            self.key, dr_key = jr.split(self.key)
            batched_model, _ = create_batched_randomized_models(
                self.mjx_model, self.num_envs, dr_key, self.dr_params
            )
        else:
            batched_model = self.mjx_model
        mjx_data_batch = jax.vmap(lambda _: mjx.make_data(self.mjx_model))(
            jnp.arange(self.num_envs)
        )
        return batched_model, mjx_data_batch

    def _sample_push(self, key, num_envs):
        k1, k2, k3, k4 = jr.split(key, 4)
        fdir = jr.normal(k1, (num_envs, 3))
        fdir = fdir / (jnp.linalg.norm(fdir, axis=-1, keepdims=True) + 1e-8)
        fmag = jr.uniform(k2, (num_envs, 1), minval=0.0, maxval=self.wrench_force_scale)
        forces  = fdir * fmag
        tdir    = jr.normal(k3, (num_envs, 3))
        tdir    = tdir / (jnp.linalg.norm(tdir, axis=-1, keepdims=True) + 1e-8)
        tmag    = jr.uniform(k4, (num_envs, 1), minval=0.0, maxval=self.wrench_torque_scale)
        torques = tdir * tmag
        return jnp.concatenate([forces, torques], axis=-1)

    def _sample_duration(self, key, num_envs, lo, hi):
        return jr.randint(key, (num_envs,), lo, hi + 1)

    def _advance_wrench(self, key, reset_mask):
        k_push, k_dur_push, k_dur_rest, k_onset = jr.split(key, 4)
        n = self.num_envs
        countdown   = self.wrench_countdown - 1
        trigger     = countdown <= 0
        new_is_push = jnp.where(trigger, jnp.logical_not(self.wrench_pushing), self.wrench_pushing)
        fresh_push  = self._sample_push(k_push, n)
        proposed_target = jnp.where(new_is_push[:, None], fresh_push, jnp.zeros((n, 6)))
        new_target      = jnp.where(trigger[:, None], proposed_target, self.wrench_target)
        push_dur    = self._sample_duration(k_dur_push, n, self.wrench_push_steps[0], self.wrench_push_steps[1])
        rest_dur    = self._sample_duration(k_dur_rest, n, self.wrench_rest_steps[0], self.wrench_rest_steps[1])
        new_dur     = jnp.where(new_is_push, push_dur, rest_dur)
        new_countdown = jnp.where(trigger, new_dur, countdown)
        a           = self.wrench_ramp_alpha
        new_current = a * self.current_wrench + (1.0 - a) * new_target
        onset       = self._sample_duration(k_onset, n, self.wrench_rest_steps[0], self.wrench_rest_steps[1])
        new_is_push   = jnp.where(reset_mask, False, new_is_push)
        new_target    = jnp.where(reset_mask[:, None], jnp.zeros((n, 6)), new_target)
        new_current   = jnp.where(reset_mask[:, None], jnp.zeros((n, 6)), new_current)
        new_countdown = jnp.where(reset_mask, onset, new_countdown)
        self.wrench_pushing   = new_is_push
        self.wrench_target    = new_target
        self.current_wrench   = new_current
        self.wrench_countdown = new_countdown

    def step(self, actions):
        self.key, step_key, wrench_key = jr.split(self.key, 3)

        xfrc = self.mjx_data_batch.xfrc_applied.at[:, self.obj_body_id, :].set(self.current_wrench)
        self.mjx_data_batch = self.mjx_data_batch.replace(xfrc_applied=xfrc)

        if self.action_ema_alpha > 0.0:
            if self.prev_smoothed_actions is None:
                self.prev_smoothed_actions = actions
            smoothed_actions = (self.action_ema_alpha * self.prev_smoothed_actions
                                + (1.0 - self.action_ema_alpha) * actions)
        else:
            smoothed_actions = actions

        prev_for_rate = self.prev_actions if self.prev_actions is not None else smoothed_actions

        _ft_ids   = self._fingertip_geom_ids
        _cube_id  = self._cube_geom_id

        def _extra_state(data):
            # positions 3D des 4 bouts de doigts → (N, 4, 3) aplati en (N, 12)
            ft = jnp.stack([
                data.site_xpos[:, self.if_tip_id, :],
                data.site_xpos[:, self.mf_tip_id, :],
                data.site_xpos[:, self.rf_tip_id, :],
                data.site_xpos[:, self.th_tip_id, :],
            ], axis=1).reshape(data.site_xpos.shape[0], 12)

            # self-collision : nb de contacts entre deux fingertips
            # (exclut les contacts avec le cube)
            def count_self_collision(g1, g2):
                # True si les deux geoms sont des fingertips (pas le cube)
                ft1 = jnp.any(jnp.isin(g1, _ft_ids))
                ft2 = jnp.any(jnp.isin(g2, _ft_ids))
                not_cube1 = jnp.logical_not(jnp.any(g1 == _cube_id))
                not_cube2 = jnp.logical_not(jnp.any(g2 == _cube_id))
                return (ft1 & ft2 & not_cube1 & not_cube2).astype(jnp.float32)

            self_coll = jax.vmap(count_self_collision)(
                data._impl.contact.geom1,
                data._impl.contact.geom2,
            ).sum(axis=-1)   # (N,) — nb de contacts doigt-doigt

            return {
                'contact_fingers':  self.current_contact_fingers,
                'pressure_fingers': self.current_pressure_fingers,
                'contact_palm':     self.current_contact_palm,
                'pressure_palm':    self.current_pressure_palm,
                'fingertip_pos':    ft,
                'self_collision':   self_coll,
            }

        (
            state, reward, reset_mask, termination, info,
            new_data, new_progress_buf, new_mjx_model, new_angvel_z_smooth
        ) = mjx_training_step(
            smoothed_actions,
            self.mjx_model_batch,
            self.mjx_data_batch,
            self.progress_buf,
            self.initial_dof_pos,
            self.reset_height_threshold,
            self.max_episode_length,
            step_key,
            self.angvel_z_smooth,
            spec=self.spec,
            reward_fn=self._reward_fn,
            termination_fn=_leap_termination_fn,
            reset_state_fn=_leap_reset_state_fn,
            control_freq_inv=self.control_freq_inv,
            use_domain_randomization=self.use_domain_randomization,
            dr_params=self.dr_params,
            base_mjx=self.base_mjx,
            action_scale=self.action_scale,
            extra_cache=self.grasp_cache,
            extra_cache_size=self.grasp_cache_size,
            extra_state_fn=_extra_state,
        )

        self.mjx_data_batch  = new_data
        self.progress_buf    = new_progress_buf
        self.mjx_model_batch = new_mjx_model
        self.angvel_z_smooth = new_angvel_z_smooth
        new_pose = new_data.qpos[:, :16]
        self.initial_dof_pos = jnp.where(
            reset_mask[:, None], new_pose, self.initial_dof_pos
        )

        if self.action_ema_alpha > 0.0:
            self.prev_smoothed_actions = jnp.where(
                reset_mask[:, None], jnp.zeros_like(smoothed_actions), smoothed_actions
            )

        wrench_mag = (
            jnp.linalg.norm(self.current_wrench[:, :3], axis=-1)
            + 0.1 * jnp.linalg.norm(self.current_wrench[:, 3:], axis=-1)
        )
        obj_motion = (
            jnp.linalg.norm(state['obj_linvel'], axis=-1)
            + jnp.linalg.norm(state['obj_angvel'], axis=-1)
        )
        wrench_resistance = self.wrench_resistance_scale * wrench_mag * jnp.exp(-3.0 * obj_motion)
        action_rate = -self.action_rate_scale * jnp.sum(
            (smoothed_actions - prev_for_rate) ** 2, axis=-1
        )
        reward = reward + wrench_resistance + action_rate

        info['wrench_resistance']   = wrench_resistance
        info['action_rate_penalty'] = action_rate
        info['wrench_mag']          = wrench_mag
        info['obj_motion']          = obj_motion

        self.prev_actions = jnp.where(
            reset_mask[:, None], jnp.zeros_like(smoothed_actions), smoothed_actions
        )
        self._advance_wrench(wrench_key, reset_mask)
        info['wrench_pushing'] = self.wrench_pushing.astype(jnp.float32)

        return (state, reward, reset_mask, termination, info,
                new_data, new_progress_buf, new_mjx_model, new_angvel_z_smooth)

    def reset(self, env_ids=None):
        if env_ids is None:
            reset_mask = jnp.ones(self.num_envs, dtype=bool)
        else:
            reset_mask = jnp.zeros(self.num_envs, dtype=bool).at[env_ids].set(True)

        self.key, subkey = jr.split(self.key)
        self.mjx_model_batch, self.mjx_data_batch = _mjx_reset_jit(
            subkey, self.mjx_model_batch, self.mjx_data_batch, reset_mask,
            self.grasp_cache,
            use_domain_randomization=self.use_domain_randomization,
            dr_params=self.dr_params,
            base_mjx=self.base_mjx,
            extra_cache_size=self.grasp_cache_size,
            spec=self.spec,
            reset_state_fn=_leap_reset_state_fn,
        )

        self.progress_buf    = jnp.where(reset_mask, 0, self.progress_buf)
        self.angvel_z_smooth = jnp.where(reset_mask, 0.0, self.angvel_z_smooth)

        self.key, wrench_key = jr.split(self.key)
        onset = self._sample_duration(
            wrench_key, self.num_envs,
            self.wrench_rest_steps[0], self.wrench_rest_steps[1]
        )
        z6 = jnp.zeros((self.num_envs, 6))
        self.wrench_pushing   = jnp.where(reset_mask, False, self.wrench_pushing)
        self.wrench_target    = jnp.where(reset_mask[:, None], z6, self.wrench_target)
        self.current_wrench   = jnp.where(reset_mask[:, None], z6, self.current_wrench)
        self.wrench_countdown = jnp.where(reset_mask, onset, self.wrench_countdown)

        if self.action_ema_alpha > 0.0 and self.prev_smoothed_actions is not None:
            self.prev_smoothed_actions = jnp.where(
                reset_mask[:, None],
                jnp.zeros_like(self.prev_smoothed_actions),
                self.prev_smoothed_actions
            )

        new_pose = self.mjx_data_batch.qpos[:, :16]
        if self.initial_dof_pos is None or env_ids is None:
            self.initial_dof_pos = new_pose
        else:
            self.initial_dof_pos = jnp.where(
                reset_mask[:, None], new_pose, self.initial_dof_pos
            )

    def get_fingertip_positions(self, mjx_data_batch=None):
        if mjx_data_batch is None:
            mjx_data_batch = self.mjx_data_batch
        return jnp.stack([
            mjx_data_batch.site_xpos[:, self.if_tip_id, :],
            mjx_data_batch.site_xpos[:, self.mf_tip_id, :],
            mjx_data_batch.site_xpos[:, self.rf_tip_id, :],
            mjx_data_batch.site_xpos[:, self.th_tip_id, :],
        ], axis=1)   # (N, 4, 3)

    def get_palm_position(self, mjx_data_batch=None):
        if mjx_data_batch is None:
            mjx_data_batch = self.mjx_data_batch
        return mjx_data_batch.xpos[:, self.palm_id, :]
