# Target Location : configs et runs

Deuxième tâche de l'article (section 7, Table 1, Fig. 3a et 5) : le personnage doit rejoindre une
position cible au sol. Environnement : `amp/location_env.py` (vérifié par
`validation/check_location_env.py`). Évaluation : `validation/eval_location.py`.

**Ce qui change par rapport à Target Heading** : le goal (direction de la cible dans le repère du
personnage et distance), la récompense de tâche, une nouvelle cible toutes les 5 à 10 s à 0–10 m,
et la fin d'épisode si le personnage est à plus de 15 m de la cible. Le reste (Φ, D, réglages de
la série 6, pieds alignés, PPO) est identique.

**Récompense** : celle du code de DeepMimic (`SceneTargetAMP.cpp`), qui a produit les chiffres de
l'article : r_G = 0,6 exp(−0,5 d²) + 0,4 r_vel, avec r_vel = exp(−4 max(0, 1 − v)²) où v est la
vitesse du centre de masse vers la cible ; r_vel = 1 à moins de 0,5 m de la cible, 0 si le
personnage s'éloigne. L'éq. 12 de l'article écrit 0,7 / 0,3 et un facteur 1 au lieu de 4 : pour
l'essayer, changer `loc_pos_reward_w`, `loc_vel_reward_w` et `loc_vel_err_scale` dans les `*_env.yaml`.

| Fichier | Contenu |
|---|---|
| `locomotion_env.yaml`, `walk_env.yaml`, `run_env.yaml` | environnement, avec les 56, 9 ou 47 clips |
| `amp_agent.yaml` | AMP, copie de `configs/run6/amp_agent.yaml` (série 6) |
| `ppo_agent.yaml` | PPO seul (baseline « No Data » de la Fig. 5), copie de `configs/run1/ppo_agent.yaml` |

## Les runs

200 M échantillons, comme la Fig. 5 de l'article (environ 2 h 30 à 22 000 échantillons/s).
Depuis la racine du projet (sous Windows, `.venv/Scripts/python`) :

```bash
python rl/train.py --agent_config configs/location/amp_agent.yaml --env_config configs/location/locomotion_env.yaml --max_samples 2e8 --out_dir outputs/rl/amp_location_locomotion
python rl/train.py --agent_config configs/location/amp_agent.yaml --env_config configs/location/walk_env.yaml --max_samples 2e8 --out_dir outputs/rl/amp_location_walk
python rl/train.py --agent_config configs/location/amp_agent.yaml --env_config configs/location/run_env.yaml --max_samples 2e8 --out_dir outputs/rl/amp_location_run
python rl/train.py --agent_config configs/location/ppo_agent.yaml --env_config configs/location/locomotion_env.yaml --max_samples 2e8 --out_dir outputs/rl/ppo_location
```

Évaluation de chaque run :

```bash
python validation/eval_location.py --run_dir outputs/rl/amp_location_locomotion
```

| Run | Ce qu'on attend |
|---|---|
| AMP Locomotion | retour normalisé autour de 0,63 (Table 1) ; marche quand la cible est proche, course quand elle est loin (courbe vitesse/distance) |
| AMP Walk | cibles atteintes en marchant ; plus lent sur les cibles lointaines |
| AMP Run | cibles atteintes en courant ; à surveiller : dépassement ou cercles autour de la cible |
| PPO seul | retour comparable (Fig. 5 : ~0,6), mouvement peu naturel |

`runs.sh` et `video_policy.py` sont écrits pour Target Heading : ne pas y ajouter ces runs pendant
que `runs.sh` tourne (bash relit le script). Les vidéos Location sont à faire.
