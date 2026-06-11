"""
test_policy.py — Tester la politique entrainees apres l'entrainement.

Usage :
    cd ~/Documents/Stage/Code/essai/tactile_regrasp-main
    MUJOCO_GL=glfw uv run python test_policy.py

Ce script :
  1. Charge le meilleur checkpoint sauvegarde dans checkpoints/
  2. Lance la politique apprise pendant N episodes
  3. Affiche les metriques (reward, contacts, chutes...)
  4. Ouvre le viewer MuJoCo pour visualiser
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import mujoco
import mujoco.viewer
import time
from pathlib import Path
from omegaconf import OmegaConf

# ── imports recurrl_jax ───────────────────────────────────────────────────────
import recurrl_jax as rjx
from recurrl_jax.model_fns import flatten_repr_model
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager

# ── imports projet ────────────────────────────────────────────────────────────
import env_wrapper as rjx_leap
import recurrl_jax.utils.wrappers as rjxw
from observation_buffer import _extract_tactile_features

# ── parametres du test ────────────────────────────────────────────────────────
N_EPISODES      = 10     # nombre d'episodes a tester
MAX_STEPS       = 500    # steps max par episode
CHECKPOINT_DIR  = "checkpoints"
SLEEP_PER_STEP  = 0.033  # secondes entre chaque step (30 fps)
SEUIL_CONTACT   = 0.001  # N


def barre(v, vmax=0.02, w=15):
    n = int(min(v / (vmax + 1e-9), 1.0) * w)
    return "█" * n + "░" * (w - n)


def load_checkpoint(checkpoint_dir):
    """Charge le meilleur checkpoint orbax."""
    ckpt_path = Path(checkpoint_dir).resolve()
    if not ckpt_path.exists():
        print(f"[ERREUR] Dossier checkpoint introuvable : {ckpt_path}")
        print("Lance d'abord train.py pour generer un checkpoint.")
        sys.exit(1)

    checkpointer = PyTreeCheckpointer()
    manager = CheckpointManager(ckpt_path, checkpointer)
    step = manager.best_step()
    if step is None:
        step = manager.latest_step()
    if step is None:
        print("[ERREUR] Aucun checkpoint trouve.")
        sys.exit(1)

    print(f"Chargement checkpoint step {step}...")
    ckpt = manager.restore(step)
    return ckpt, step


def make_eval_env(num_envs=1):
    """Cree l'environnement de test (1 seul env, pas de normalisation update)."""
    env = rjx_leap.LeapHandGymWrapper(
        num_envs=num_envs,
        use_domain_randomization=False,
        normalize_obs=True,
        action_scale=0.6,
        action_ema_alpha=0.6,
        grasp_cache_path=None,
        fingertip_proximity_scale=5.0,
        fingertip_contact_scale=2.0,
        self_collision_scale=1.0,
        reward_scale=0.01,
        update_norm_stats=False,   # ne pas mettre a jour les stats pendant le test
    )
    return rjxw.SqueezeWrapper(env)


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          TEST DE LA POLITIQUE ENTRAINEE                  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── charger le checkpoint ─────────────────────────────────────────────────
    ckpt, step = load_checkpoint(CHECKPOINT_DIR)
    params = ckpt['params']

    # restaurer les stats de normalisation si disponibles
    obs_mean = ckpt.get('obs_mean', None)
    obs_var  = ckpt.get('obs_var', None)

    # ── creer l'environnement de test ─────────────────────────────────────────
    print("Creation de l'environnement...")
    env = make_eval_env(num_envs=1)

    if obs_mean is not None and hasattr(env, 'env'):
        base = env.env
        while hasattr(base, 'env'):
            base = base.env
        if hasattr(base, 'running_mean_std'):
            base.running_mean_std.mean = jnp.array(obs_mean)
            base.running_mean_std.var  = jnp.array(obs_var)
            print("Stats de normalisation restaurees.")

    # ── creer le modele acteur ────────────────────────────────────────────────
    print("Creation du modele PPO...")
    key = jr.PRNGKey(0)

    # On reconstruit le meme modele que pendant l'entrainement
    # Les dimensions doivent correspondre exactement
    trainer = rjx.Trainer(
        env_factory=lambda ec, tc, gc: rjxw.VectorEpisodeStatisticsWrapper(
            make_eval_env(num_envs=1)
        ),
        eval_env_factory=lambda ec, tc, gc, te: make_eval_env(num_envs=1),
        repr_fn=flatten_repr_model(),
        is_continuous=True,
        video_render_fn=None,
        global_args=OmegaConf.create({
            'seed': 0, 'steps': 1, 'log_interval': 1,
            'eval_episodes': 1, 'eval_interval': 999999999,
            'use_wandb': False, 'checkpoint_dir': None,
            'save_best_only': False, 'render_videos': False,
        }),
        trainer_config=OmegaConf.create({
            'agent': 'ppo', 'd_actor': 256, 'd_critic': 256,
            'num_envs': 1, 'rollout_len': 32, 'sequence_length': 8,
            'gamma': 0.99, 'gae_lambda': 0.97, 'num_minibatches': 8,
            'update_epochs': 6, 'norm_adv': True, 'clip_coef': 0.2,
            'history_len': 3, 'vf_coef': 0.5, 'max_grad_norm': 0.5,
            'ent_coef': {'initial': 0.005, 'final': 0.0001,
                         'max_decay_steps': 80000000, 'power': 1},
            'adaptive_lr': {'enabled': True, 'kl_threshold': 0.02,
                            'lr_min': 1e-6, 'lr_max': 0.01},
            'seq_model': {'name': 'gru', 'd_model': 256, 'n_layers': 1,
                          'reset_hidden_on_terminate': True},
        }),
        env_config=OmegaConf.create({}),
        seed=0,
        key=key,
        wandb_run=None,
    )

    # Injecter les parametres appris
    trainer.agent._params = params
    print(f"Politique chargee (step {step})")

    # ── viewer MuJoCo ─────────────────────────────────────────────────────────
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    mj_model = base_env.env.mj_model
    mj_data  = mujoco.MjData(mj_model)

    # ── boucle de test ────────────────────────────────────────────────────────
    episode_rewards   = []
    episode_lengths   = []
    episode_contacts  = []
    episode_drops     = []

    print(f"\nLancement de {N_EPISODES} episodes de test...")
    print("Ferme la fenetre MuJoCo pour arreter.\n")

    with mujoco.viewer.launch_passive(mj_model, mj_data) as v:
        v.cam.azimuth   = 180
        v.cam.elevation = -25
        v.cam.distance  = 0.45
        v.cam.lookat[:] = [0.10, 0.0, 0.05]

        for ep in range(N_EPISODES):
            if not v.is_running():
                break

            obs, _ = env.reset()
            obs = jnp.array(obs)

            ep_reward     = 0.0
            ep_steps      = 0
            ep_contacts   = 0
            ep_drops      = 0
            hidden        = None   # etat cache GRU

            print(f"Episode {ep+1}/{N_EPISODES}")

            for step_i in range(MAX_STEPS):
                if not v.is_running():
                    break

                # Obtenir l'action de la politique
                action, hidden = trainer.agent.act(obs[None], hidden)
                action = jnp.squeeze(action)

                # Avancer la simulation
                obs, reward, terminated, truncated, info = env.step(np.array(action))
                obs = jnp.array(obs)

                ep_reward   += float(reward)
                ep_steps    += 1

                # Stats tactiles
                base = env
                while hasattr(base, 'env'):
                    base = base.env
                tactile = np.array(base.env._tactile_obs[0])
                p_fingers = np.array([
                    tactile[0:3].mean(), tactile[3:6].mean(),
                    tactile[6:9].mean(), tactile[9:12].mean(),
                ])
                n_contact = (p_fingers > SEUIL_CONTACT).sum()
                ep_contacts += n_contact

                cube_z = float(base.env.mjx_data_batch.qpos[0, 18])
                if cube_z < -0.05:
                    ep_drops += 1

                # Mettre a jour le viewer
                mj_data.qpos[:] = np.array(base.env.mjx_data_batch.qpos[0])
                mj_data.qvel[:] = np.array(base.env.mjx_data_batch.qvel[0])
                mujoco.mj_forward(mj_model, mj_data)
                v.sync()

                # Affichage terminal
                os.system('clear')
                print(f"Episode {ep+1}/{N_EPISODES}  Step {step_i+1}/{MAX_STEPS}")
                print(f"  Reward cumulé  : {ep_reward:>8.3f}")
                print(f"  Cube z         : {cube_z:>6.3f} m")
                print(f"  Doigts contact : {n_contact}/4")
                print()
                print("  Pression par doigt :")
                for nom, p in zip(["Index","Majeur","Annulaire","Pouce"], p_fingers):
                    flag = " ◄" if p > SEUIL_CONTACT else ""
                    print(f"    {nom:<12} {barre(p)} {p:.4f} N{flag}")
                print()
                print(f"  Episodes précédents :")
                for i, (r, l, c) in enumerate(zip(episode_rewards, episode_lengths, episode_contacts)):
                    avg_c = c / max(l, 1)
                    print(f"    Ep {i+1} : reward={r:>7.2f}  longueur={l:>4}  contacts moy={avg_c:.2f}")

                time.sleep(SLEEP_PER_STEP)

                if terminated or truncated:
                    break

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_steps)
            episode_contacts.append(ep_contacts)
            episode_drops.append(ep_drops)

            print(f"\n  → Episode {ep+1} terminé : reward={ep_reward:.2f}  steps={ep_steps}  drops={ep_drops}")
            time.sleep(1.0)

    # ── résumé final ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RÉSUMÉ DU TEST")
    print("="*60)
    print(f"  Reward moyen     : {np.mean(episode_rewards):>8.3f} ± {np.std(episode_rewards):.3f}")
    print(f"  Longueur moyenne : {np.mean(episode_lengths):>8.1f} steps")
    print(f"  Chutes moyennes  : {np.mean(episode_drops):>8.1f} par épisode")
    avg_c_per_step = np.sum(episode_contacts) / max(np.sum(episode_lengths), 1)
    print(f"  Contacts FSR moy : {avg_c_per_step:>8.3f} doigts/step")
    print("="*60)


if __name__ == "__main__":
    main()
