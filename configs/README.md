# Configs

Un dossier par série de runs : `configs/runN/` contient tout ce qu'il faut pour relancer les runs
`outputs/rl/<tâche>_N` (série 1 : sans suffixe, `amp_walk`, `amp_run`, `amp_locomotion`, `ppo_heading`).

```
configs/
├── runN/
│   ├── amp_agent.yaml         agent AMP de la série (PPO + discriminateur D)
│   ├── walk_env.yaml          environnement Target Heading, clips Walk (9)
│   ├── run_env.yaml           idem, clips Run (47)
│   └── locomotion_env.yaml    idem, Walk + Run (56 clips)
├── run1/ppo_agent.yaml        PPO seul (r_G seulement), run ppo_heading
├── datasets/                  listes de clips Walk et Run (lues par les *_env.yaml)
├── ppo_heading_humanoid_args.txt   run PPO via le PPO de MimicKit (amp/run.py, outputs/phase3/)
├── amp_heading_humanoid_args.txt   run AMP via notre agent sur le PPO de MimicKit (amp/run.py)
└── newton_test_nopd_240hz.yaml     moteur sans PD à 240 Hz, pour validation/check_heading_env.py (test 2)
```

Lancer un run (depuis la racine du projet, venv activé) :

```bash
python rl/train.py --agent_config configs/run5/amp_agent.yaml --env_config configs/run5/run_env.yaml --out_dir outputs/rl/amp_run_5
```

`runs.sh` enchaîne les runs (et leur évaluation, `validation/eval_policy.py` et `validation/video_policy.py`). Chaque run recopie ses deux configs dans son dossier de
sortie (`agent_config.yaml`, `env_config.yaml`) : c'est la référence exacte de ce qui a tourné.

Les `*_env.yaml` des séries 2 et 3 sont des copies de ceux de `run1/`, et ceux des séries 5 et 6 des
copies de ceux de `run4/`. La première ligne de chaque copie l'indique.

## Différences entre les séries

Chaque série part de la série 2, sauf la série 1, qui est la référence. Tout ce qui n'est pas dans le
tableau est identique : PPO, réseaux, D en LSGAN avec pénalité de gradient 10, replay buffer, 300 M
échantillons.

| Série | `disc_steps` | pas de la politique | régularisation de D | pieds alignés | entrée de D | Résultat sur Run |
|---|---|---|---|---|---|---|
| 1 | 32 | 5e-5 | non | non | normalisée sur la mocap seule | r_S s'effondre (≈ 0,05) |
| 2 | **8** | 5e-5 | non | non | mocap seule | effondrement un peu retardé |
| 3 | 8 | 5e-5 | **weight decay 1e-4 + pénalité 0,01** | non | mocap seule | comme la série 2 |
| 4 | 8 | **4e-6** | non | **oui** | mocap seule | tâche beaucoup plus lente, même D (arrêté à 85 M) |
| 5 | 8 | 5e-5 | non | oui | mocap seule, **entrées constantes masquées** | en cours |
| 6 | 8 | 5e-5 | non | oui | **mocap + agent (MimicKit)** | à venir |

Ce que règle chaque paramètre :

- **`disc_steps`** (`amp_agent.yaml`) : nombre de mises à jour de D par itération. Avec 32, D voit chaque
  paire de l'agent du replay buffer environ 30 fois, et avec 8, environ 8 fois.
- **Pas de la politique** (`actor_optimizer.learning_rate`) : 5e-5 est la valeur de MimicKit, 4e-6 celle
  de l'article (Table 4).
- **Régularisation de D** (`disc_optimizer.weight_decay`, `disc_logit_reg`) : celle de l'AMP de MimicKit
  (`amp_agent.py`). Elle n'est pas dans l'article.
- **Pieds alignés** (`align_mocap_feet`, `foot_contact_height` dans les `*_env.yaml`) : chaque clip est
  décalé verticalement pour que ses pieds d'appui soient au niveau de contact de la simulation. Les
  pieds de la mocap Run étaient 2 à 3 cm sous le sol. Code dans `amp/foot_align.py`, tests dans
  `validation/check_foot_align.py`. Ce n'est pas dans l'article.
- **Entrées constantes masquées** (`disc_mask_constant`) : les entrées de Φ dont l'écart-type sur la
  mocap est inférieur à 1e-3 sont mises à 0 pour D, mocap comme agent. Pour Run, cela concerne le cou :
  il vaut exactement 0 dans les 47 clips, ce qui donnait à D un indice parfait. Pour Walk et Locomotion,
  seule une entrée est masquée, et elle est constante aussi pour l'agent. Ce n'est pas dans l'article.
- **Normalisation mocap + agent** (`disc_norm_agent`) : la moyenne et l'écart-type de l'entrée de D sont
  mis à jour à chaque itération avec les paires de l'agent et autant de paires mocap, comme dans
  MimicKit. Ils sont figés après `normalizer_samples` (1e8). Ce n'est pas dans l'article, qui ne
  décrit pas cette normalisation.

Ces options sont désactivées par défaut dans `rl/amp.py`. Les configs des séries précédentes donnent
donc toujours le même calcul.

## Environnements

Les trois `*_env.yaml` d'une série ne diffèrent que par `motion_file` (le jeu de clips). Ceux des
séries 4 à 6 ajoutent `align_mocap_feet: True` et `foot_contact_height`, qui vaut −0,013 m pour Walk
et Locomotion, et −0,017 m pour Run. Ces valeurs sont mesurées sur la démarche simulée par
`validation/check_foot_align.py`, test 1. La base commune est `run1/locomotion_env.yaml`. Les différences
avec l'environnement de MimicKit y sont marquées `[CHANGED]`.
