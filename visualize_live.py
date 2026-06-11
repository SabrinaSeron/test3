"""
visualize_live.py — Visualisation en temps reel pendant l'entrainement.

Terminal 1 : uv run python train.py
Terminal 2 : MUJOCO_GL=glfw uv run python visualize_live.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
import mujoco
import mujoco.viewer

SHARED_FILE = "/tmp/tactile_live_state.npy"
SEUIL       = 0.001   # N — seuil de contact FSR

DOIGT_NOMS  = ["Index", "Majeur", "Annulaire", "Pouce", "Paume"]


def barre(v, vmax=0.02, w=18):
    n = int(min(v / (vmax + 1e-9), 1.0) * w)
    return "█" * n + "░" * (w - n)


def afficher(state):
    tactile       = state.get('tactile_obs', np.zeros(13))
    step          = state.get('step', 0)
    reward        = state.get('reward', 0.0)
    alive         = state.get('alive_reward', 0.0)
    proximity     = state.get('proximity_bonus', 0.0)
    ft_contact    = state.get('fingertip_contact', 0.0)
    self_coll     = state.get('self_collision', 0.0)
    drop          = state.get('drop_penalty', 0.0)
    n_contact     = state.get('n_fingers_contact', 0)
    cube_z        = state.get('cube_z', 0.0)

    os.system('clear')
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     VISUALISATION TACTILE EN TEMPS RÉEL — LEAP Hand     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Step : {int(step):>8}   Reward total : {reward:>8.4f}")
    print(f"  Cube z : {cube_z:>6.3f} m   Doigts en contact FSR : {int(n_contact)}/4")
    print()

    # ── Pression par doigt ────────────────────────────────────────────────────
    print("  ── PRESSION PAR DOIGT (FSR) ────────────────────────────────")
    pressions = [
        tactile[0:3].mean(),
        tactile[3:6].mean(),
        tactile[6:9].mean(),
        tactile[9:12].mean(),
        tactile[12],
    ]
    for nom, p in zip(DOIGT_NOMS, pressions):
        flag = " ◄ CONTACT" if p > SEUIL else ""
        print(f"  {nom:<12} {barre(p)} {p:.4f} N{flag}")
    print()

    # ── Termes de récompense ──────────────────────────────────────────────────
    print("  ── RÉCOMPENSE ──────────────────────────────────────────────")
    termes = [
        ("alive_reward",       alive,      "+2.0 = cube tenu"),
        ("proximity_bonus",    proximity,  "doigts proches du cube"),
        ("fingertip_contact",  ft_contact, "bout de doigt touche le cube"),
        ("self_collision pen", -self_coll, "doigts qui se touchent"),
        ("drop_penalty",       drop,       "-1000 si chute"),
        ("reward TOTAL",       reward,     ""),
    ]
    for nom, val, desc in termes:
        bar = "+" if val >= 0 else ""
        note = f"  ← {desc}" if desc else ""
        print(f"  {nom:<22} {bar}{val:>8.4f}{note}")
    print()
    print("  [Ctrl+C pour arrêter]")


def main():
    print("Chargement MuJoCo...")
    mj_model = mujoco.MjModel.from_xml_path("xmls/scene_mjx.xml")
    mj_data  = mujoco.MjData(mj_model)

    print(f"En attente de {SHARED_FILE}...")
    print("Lance train.py dans un autre terminal.")

    with mujoco.viewer.launch_passive(mj_model, mj_data) as v:
        v.cam.azimuth   = 180
        v.cam.elevation = -25
        v.cam.distance  = 0.45
        v.cam.lookat[:] = [0.10, 0.0, 0.05]

        last_step = -1
        while v.is_running():
            try:
                state = np.load(SHARED_FILE, allow_pickle=True).item()

                if 'qpos' in state and 'qvel' in state:
                    mj_data.qpos[:len(state['qpos'])] = state['qpos']
                    mj_data.qvel[:len(state['qvel'])] = state['qvel']
                    mujoco.mj_forward(mj_model, mj_data)
                    v.sync()

                if state.get('step', 0) != last_step:
                    afficher(state)
                    last_step = state.get('step', 0)

            except FileNotFoundError:
                print(f"\rEn attente de {SHARED_FILE}...", end="")
            except Exception as e:
                print(f"\rErreur : {e}", end="")

            time.sleep(0.033)


if __name__ == "__main__":
    main()
