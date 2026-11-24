# Validation

Scripts de test et d'évaluation, à lancer depuis la racine du projet, venv `.venv` activé. Le préfixe
dit à quoi sert un script : `check_` teste du code, `eval_` évalue un run entraîné, `video_` produit
des vidéos. Chaque script décrit ses tests et ses options dans sa docstring (`python validation/<script>.py --help`).

| Script | Rôle | Ancien nom |
|---|---|---|
| `check_mimickit_install.py` | installation de MimicKit sous Newton, comparaison à leur log (phase 0) | `phase0.py` |
| `check_disc_features.py` | features Φ de D et transitions expertes (phase 1) | `phase1.py` |
| `video_mocap_clip.py` | vidéo d'un clip mocap joué dans Newton | `phase1_view_clip.py` |
| `check_heading_env.py` | environnement Target Heading, sans apprentissage (phase 2) | `phase2.py` |
| `video_dummy_policies.py` | vidéos de politiques factices : chute, suivi de la mocap | `phase2_videos.py` |
| `eval_policy.py` | **évaluation d'un run** : courbe, retour normalisé (Table 1), vitesse atteinte, chutes | `phase3.py` |
| `video_policy.py` | **vidéos d'un run** dans `<run>/videos/` | `phase3_videos.py` |
| `check_amp.py` | features, D et équations de l'article, sur notre AMP basé sur MimicKit (phase 4) | `phase4.py` |
| `check_rl_vs_mimickit.py` | notre `rl/` comparé à MimicKit, brique par brique | `rl_check.py` |
| `check_foot_align.py` | alignement vertical des pieds de la mocap (`amp/foot_align.py`) | `foot_align.py` |
| `load_envs.py` | module commun, construit les environnements une fois par processus | |

Bilans écrits : `check_mimickit_install.md`, `check_disc_features.md` et `check_heading_env.md`.
Dossiers de sortie : `outputs/mimickit_install/` (installation), `outputs/mocap/` (clips, vitesses),
`outputs/heading_env/` (politiques factices), `outputs/mimickit/` (runs via le PPO de MimicKit).

## Après un run d'entraînement

`runs.sh` lance automatiquement l'évaluation et les vidéos après chaque run. À la main :

```bash
python validation/eval_policy.py --run_dir outputs/rl/amp_run_5 > outputs/rl/amp_run_5/eval_policy.txt
python validation/video_policy.py --run_dir outputs/rl/amp_run_5               # schedule.mp4, scripted.mp4
python validation/video_policy.py --run_dir outputs/rl/amp_run_5 --fixed_goal  # fixed_goal.mp4
```

ou `bash runs.sh eval amp_run_5`. Les seuils de réussite de `eval_policy.py` ont été fixés pour le PPO
seul. Pour AMP, on lit surtout les chiffres. Après une modification de `rl/`, lancer
`check_rl_vs_mimickit.py`.
