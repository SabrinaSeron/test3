"""
rewards.py — fonctions de recompense pour la LEAP Hand.

Progression pedagogique de l'apprentissage :
  1. proximity_bonus    : approcher les doigts du cube
  2. fingertip_contact  : toucher le cube avec les bouts de doigts
  3. contact_bonus      : plusieurs doigts en contact simultanément
  4. pressure_bonus     : appuyer à la bonne intensité
  5. palm_bonus         : contact paume (doux)

Penalités :
  self_collision_penalty : doigts qui se touchent entre eux
  slip_penalty           : cube qui bouge/tourne
  torque_penalty         : effort moteur excessif
  low_pressure_penalty   : contact trop léger
  high_pressure_penalty  : pression trop forte
  drop_term              : chute du cube (-1000)
"""

import jax
import jax.numpy as jnp

P_LOW  = 0.001   # N — pression minimale acceptable
P_HIGH = 0.01    # N — pression maximale acceptable


def compute_fingertip_proximity_bonus(
    fingertip_pos: jax.Array,   # (N, 12) — aplati depuis (N, 4, 3)
    obj_pos:       jax.Array,   # (N, 3)
    scale:         float = 5.0,
) -> jax.Array:                 # (N,)
    """
    Bonus de proximite : recompense les doigts proches du cube.
    Utilise distance^2 pour eviter sqrt(0) qui produit des NaN.
    Formule : scale * exp(-distance_moyenne^2)
    → max quand les doigts touchent, diminue exponentiellement
    """
    fingertip_pos = jnp.nan_to_num(fingertip_pos, nan=0.0, posinf=0.0, neginf=0.0)
    obj_pos       = jnp.nan_to_num(obj_pos,       nan=0.0, posinf=0.0, neginf=0.0)

    # (N, 12) → (N, 4, 3)
    ft = fingertip_pos.reshape(fingertip_pos.shape[0], 4, 3)

    # vecteur doigt → cube : (N, 4, 3)
    diff = ft - obj_pos[:, None, :]

    # distance^2 par doigt puis moyenne : (N,)
    dist_sq = jnp.sum(diff ** 2, axis=-1).mean(axis=-1)

    return scale * jnp.exp(-dist_sq)


def compute_fingertip_contact_bonus(
    fingertip_contact: jax.Array,   # (N,) bool — True si au moins 1 fingertip touche le cube
    scale: float = 2.0,
) -> jax.Array:                     # (N,)
    """
    Bonus direct quand un bout de doigt touche le cube.
    fingertip_contact vient de state['contact'] déjà calculé par mjx_env.py.
    C'est un signal binaire clair — le doigt touche ou pas.
    """
    return scale * fingertip_contact.astype(jnp.float32)


def compute_self_collision_penalty(
    self_collision: jax.Array,   # (N,) float — nombre de contacts doigt-doigt
    scale: float = 1.0,
) -> jax.Array:                  # (N,)
    """
    Penalite quand les doigts se touchent entre eux.
    On a vu dans le viewer que les doigts se ferment et se touchent
    au lieu de serrer le cube — ce terme corrige ce comportement.
    """
    return -scale * self_collision


def compute_tactile_reward(
    contact_fingers:   jax.Array,   # (N, 4)
    pressure_fingers:  jax.Array,   # (N, 4)
    contact_palm:      jax.Array,   # (N, 1)
    pressure_palm:     jax.Array,   # (N, 1)
    contact_bonus_scale:    float = 1.0,
    pressure_bonus_scale:   float = 0.5,
    palm_bonus_scale:       float = 0.3,
    low_pressure_scale:     float = 0.5,
    high_pressure_scale:    float = 0.2,
) -> tuple:
    """
    Termes de recompense tactiles FSR.
    Appelé seulement quand les capteurs détectent quelque chose.
    """
    n_contacts = contact_fingers.sum(axis=-1)   # (N,)

    # BONUS : plusieurs doigts en contact simultanément
    # 0 ou 1 doigt → 0, 2 doigts → +1, 3 → +2, 4 → +3
    contact_bonus = contact_bonus_scale * jnp.maximum(n_contacts - 1.0, 0.0)

    # BONUS : pression dans la zone cible [P_LOW, P_HIGH]
    in_range = jnp.logical_and(
        pressure_fingers >= P_LOW,
        pressure_fingers <= P_HIGH,
    ).astype(jnp.float32)
    pressure_bonus = pressure_bonus_scale * (in_range * contact_fingers).sum(axis=-1)

    # BONUS : contact paume (doux)
    palm_bonus = palm_bonus_scale * contact_palm.reshape(-1)

    # PENALITE : contact détecté mais pression trop faible → instable
    too_low = jnp.logical_and(
        contact_fingers > 0.5,
        pressure_fingers < P_LOW,
    ).astype(jnp.float32)
    low_pressure_penalty = -low_pressure_scale * too_low.sum(axis=-1)

    # PENALITE : pression trop forte → effort inutile
    excess = jnp.maximum(
        jnp.nan_to_num(pressure_fingers, nan=0.0) - P_HIGH, 0.0
    )
    high_pressure_penalty = -high_pressure_scale * excess.sum(axis=-1)

    reward_tactile = (
        contact_bonus
        + pressure_bonus
        + palm_bonus
        + low_pressure_penalty
        + high_pressure_penalty
    )

    info = {
        'contact_bonus':          contact_bonus,
        'pressure_bonus':         pressure_bonus,
        'palm_bonus':             palm_bonus,
        'low_pressure_penalty':   low_pressure_penalty,
        'high_pressure_penalty':  high_pressure_penalty,
        'n_fingers_contact':      n_contacts,
    }
    return reward_tactile, info


def compute_catching_reward(
    obj_linvel:        jax.Array,   # (N, 3)
    obj_angvel:        jax.Array,   # (N, 3)
    obj_pos:           jax.Array,   # (N, 3)
    torques:           jax.Array,   # (N, 16)
    contact_fingers:   jax.Array,   # (N, 4) — depuis FSR
    pressure_fingers:  jax.Array,   # (N, 4)
    contact_palm:      jax.Array,   # (N, 1)
    pressure_palm:     jax.Array,   # (N, 1)
    fingertip_pos:     jax.Array,   # (N, 12) — positions bouts de doigts aplaties
    fingertip_contact: jax.Array,   # (N,) — state['contact'] : fingertip touche le cube
    self_collision:    jax.Array,   # (N,) — nb contacts doigt-doigt
    reset_height_threshold:       float = -0.05,
    slip_vel_scale:               float = 0.1,
    torque_scale:                 float = 0.0001,
    alive_bonus:                  float = 2.0,
    drop_penalty:                 float = -1000.0,
    fingertip_proximity_scale:    float = 5.0,
    fingertip_contact_scale:      float = 2.0,
    self_collision_scale:         float = 1.0,
    contact_bonus_scale:          float = 1.0,
    pressure_bonus_scale:         float = 0.5,
    palm_bonus_scale:             float = 0.3,
    low_pressure_scale:           float = 0.5,
    high_pressure_scale:          float = 0.2,
) -> tuple:

    # protection NaN
    obj_linvel = jnp.nan_to_num(obj_linvel, nan=0.0, posinf=0.0, neginf=0.0)
    obj_angvel = jnp.nan_to_num(obj_angvel, nan=0.0, posinf=0.0, neginf=0.0)
    torques    = jnp.nan_to_num(torques,    nan=0.0, posinf=0.0, neginf=0.0)

    # ── termes d'Eduardo (inchangés) ─────────────────────────────────────────
    linvel_mag   = jnp.linalg.norm(obj_linvel, axis=-1)
    angvel_mag   = jnp.minimum(jnp.linalg.norm(obj_angvel, axis=-1), 10.0)
    slip_penalty = -slip_vel_scale * (linvel_mag + 0.5 * angvel_mag)

    torque_penalty = -torque_scale * jnp.sum(torques ** 2, axis=-1)

    alive      = (obj_pos[:, 2] >= reset_height_threshold).astype(jnp.float32)
    alive_reward = alive_bonus * alive
    drop_term    = (1.0 - alive) * drop_penalty

    # ── nouveaux termes de guidage ────────────────────────────────────────────

    # 1. Proximité : approcher les doigts du cube
    proximity_bonus = compute_fingertip_proximity_bonus(
        fingertip_pos, obj_pos, scale=fingertip_proximity_scale
    )

    # 2. Contact direct fingertip → cube (signal binaire fort)
    ft_contact_bonus = compute_fingertip_contact_bonus(
        fingertip_contact, scale=fingertip_contact_scale
    )

    # 3. Pénalité self-collision : doigts qui se touchent entre eux
    self_coll_penalty = compute_self_collision_penalty(
        self_collision, scale=self_collision_scale
    )

    # ── termes tactiles FSR ───────────────────────────────────────────────────
    reward_tactile, tactile_info = compute_tactile_reward(
        contact_fingers, pressure_fingers, contact_palm, pressure_palm,
        contact_bonus_scale, pressure_bonus_scale, palm_bonus_scale,
        low_pressure_scale, high_pressure_scale,
    )

    # ── total ─────────────────────────────────────────────────────────────────
    total = (
        alive_reward
        + slip_penalty
        + torque_penalty
        + drop_term
        + proximity_bonus
        + ft_contact_bonus
        + self_coll_penalty
        + reward_tactile
    )

    info = {
        'alive_reward':          alive_reward,
        'slip_penalty':          slip_penalty,
        'torque_penalty':        torque_penalty,
        'drop_penalty':          drop_term,
        'proximity_bonus':       proximity_bonus,
        'fingertip_contact':     ft_contact_bonus,
        'self_collision_penalty': self_coll_penalty,
        'obj_linvel_mag':        linvel_mag,
        'obj_angvel_mag':        angvel_mag,
        **tactile_info,
    }
    return total, info


def check_termination(
    object_pos:             jax.Array,
    progress_buf:           jax.Array,
    max_episode_length:     int = 500,
    reset_height_threshold: float = -0.05,
) -> tuple:
    object_fallen     = object_pos[:, 2] < reset_height_threshold
    has_nan           = jnp.any(jnp.isnan(object_pos), axis=-1)
    has_inf           = jnp.any(jnp.isinf(object_pos), axis=-1)
    extreme           = jnp.any(jnp.abs(object_pos) > 10.0, axis=-1)
    physics_explosion = has_nan | has_inf | extreme
    termination       = object_fallen | physics_explosion
    reset_mask        = termination | (progress_buf >= max_episode_length)
    return reset_mask, termination
