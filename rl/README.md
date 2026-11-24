# `rl/` : notre PPO et notre AMP, et les entraînements à lancer

`rl/` réimplémente toute la partie apprentissage qu'on prenait dans `mimickit/learning/` : réseaux,
politique gaussienne, normalisation, buffer, retours TD(λ) et GAE, PPO, boucle d'entraînement, et
AMP par-dessus (`rl/amp.py`). De MimicKit, on ne garde que le « monde » : Newton, l'environnement
(`amp/heading_env.py`), la mocap et le squelette.

`validation/check_rl_vs_mimickit.py` vérifie que, avec les mêmes poids et les mêmes données, notre code donne
**exactement** les mêmes sorties que MimicKit (réseaux, normalisation, TD(λ), une itération de PPO,
une itération d'AMP). Ce qui reste à vérifier, c'est qu'il **apprend** pareil sur un vrai
entraînement : c'est le run 1.

| Fichier | Rôle |
|---|---|
| `nets.py` | MLP 1024-512 ReLU |
| `models.py` | π gaussien (σ fixe) et V |
| `normalizer.py` | moyenne et écart-type courants des observations |
| `buffer.py` | les 32 pas × num_envs d'une itération |
| `returns.py` | retours TD(λ), d'où les avantages GAE |
| `ppo.py` | PPO : choix des actions, pertes, mises à jour |
| `train.py` | boucle d'entraînement, tests, logs, sauvegardes, point d'entrée |
| `amp.py` | discriminateur D, éq. 6-8, replay buffer, récompense de style |

## Avant de lancer

1. Récupérer le code (`git pull`).
2. Depuis la racine du projet, vérifier que tout marche sur ta machine (2 à 3 min, 64 envs) :
   ```bash
   .venv/Scripts/python validation/check_rl_vs_mimickit.py
   ```
   Les 5 tests doivent être OK.
3. Les commandes ci-dessous sont à lancer **depuis la racine du projet**. Sous Linux, remplacer
   `.venv/Scripts/python` par `.venv/bin/python`.

Tous les runs écrivent dans `--out_dir` : `log.txt` (une ligne toutes les 50 itérations, même
format que MimicKit), `model.pt` (le dernier modèle), `int_models/` (un modèle toutes les 50
itérations) et une copie des configs. Pendant le run, une ligne par itération s'affiche :
échantillons, retour d'entraînement, retour et durée des épisodes de test, échantillons/s.

Il n'y a pas de reprise : un run interrompu est à relancer depuis le début.

## Run 1 : PPO seul (à faire en premier)

**But.** Vérifier que notre PPO apprend comme celui de MimicKit, en refaisant exactement le run de
la phase 3 (récompense de tâche seule, mêmes réglages, même graine) avec `rl/` au lieu de MimicKit.

```bash
.venv/Scripts/python rl/train.py --agent_config configs/run1/ppo_agent.yaml --out_dir outputs/rl/ppo_heading
```

300 M échantillons par défaut. La phase 3 avait pris 3,6 h sur ta 3070.

**Ce qu'on attend** : la même courbe que `outputs/mimickit/ppo_heading/log.txt`, au bruit près
(les tirages aléatoires ne sont pas les mêmes). Repères de la phase 3 :

| Échantillons | Durée des épisodes de test | Retour de test |
|---|---|---|
| 66 M | 36 pas | 13 |
| 98 M | 501 pas | 367 |
| 300 M | 589 pas (19,6 s sur 20) | 548 |

Pour tracer la courbe : `validation/eval_policy.py --run_dir outputs/rl/ppo_heading --tests 1`, qui écrit
`learning_curve.png` dans le dossier du run (les tests 2 à 4 de `eval_policy.py` chargent un modèle
MimicKit et ne marchent pas encore avec les nôtres).

**Si la courbe ne décolle pas vers 80-100 M** ou finit nettement plus bas, inutile de lancer la suite :
il y a une différence à trouver entre notre PPO et celui de MimicKit.

## Runs 2 à 4 : AMP sur Walk, Run et Locomotion

**But.** Les résultats de l'article (section 8.2, Fig. 3 et 4) : la même tâche Target Heading, avec
en plus le discriminateur, qui pousse le personnage à bouger comme dans la mocap. Chaque run utilise
un jeu de clips différent, pour le discriminateur et pour les démarrages d'épisode :

| Run | Jeu de données | Ce que l'article montre |
|---|---|---|
| Walk | 9 clips de marche (229 s) | une marche naturelle vers 1 m/s ; la politique n'atteint pas les vitesses élevées |
| Run | 47 clips de course (204 s) | une course, de 1,5 à 5 m/s |
| Locomotion | les 56 (Walk + Run) | marche vers 1 m/s, jog vers 2,5 m/s, course vers 4,5 m/s selon v* ; le personnage s'incline dans les virages |

```bash
.venv/Scripts/python rl/train.py --agent_config configs/run1/amp_agent.yaml --env_config configs/run1/walk_env.yaml --out_dir outputs/rl/amp_walk
.venv/Scripts/python rl/train.py --agent_config configs/run1/amp_agent.yaml --env_config configs/run1/run_env.yaml --out_dir outputs/rl/amp_run
.venv/Scripts/python rl/train.py --agent_config configs/run1/amp_agent.yaml --env_config configs/run1/locomotion_env.yaml --out_dir outputs/rl/amp_locomotion
```

Durée : non mesurée, plus lente que PPO seul (D est mis à jour 32 fois par itération). Regarder les
échantillons/s affichés après quelques minutes pour l'estimer. 300 M par défaut (`--max_samples`
pour changer).

**Ce qu'il faut regarder dans `log.txt`**, en plus de `Test_Return` (qui reste la somme de r_G
seule, donc comparable au run 1) :

| Colonne | Ce qui est sain | Ce qui signale un problème |
|---|---|---|
| `Disc_Acc_Mocap`, `Disc_Acc_Agent` | entre ~0,55 et ~0,95 : D distingue, sans gagner complètement | collées à 1,0 : D a gagné, r_S tombe à 0 ; collées à 0,5 : D n'apprend rien |
| `Style_Reward_Mean` (r_S moyen) | baisse au début (D apprend), puis remonte quand le mouvement devient naturel | reste proche de 0 |
| `Disc_Grad_Penalty` | petite, stable | qui explose |
| `Task_Reward_Mean` (r_G moyen) | monte comme en PPO, sans doute plus lentement | reste plate alors que le run 1 décollait |

On attend un retour de tâche un peu plus bas qu'en PPO seul (le style a un prix), mais un mouvement
bien plus naturel : c'est à vérifier à l'œil.

## Tester une politique entraînée

```bash
.venv/Scripts/python rl/train.py --mode test --model_file outputs/rl/amp_walk/model.pt --env_config configs/run1/walk_env.yaml --num_envs 64
```

Affiche le retour et la durée moyens des épisodes, avec l'action moyenne (sans bruit). Les scripts
de vidéo (`validation/video_policy.py`) chargent pour l'instant des modèles MimicKit : à adapter.

## Si ça plante

- Mémoire GPU insuffisante : baisser `--num_envs` (par exemple 2048). Les lots de PPO et de D sont
  proportionnels à num_envs, l'apprentissage reste comparable mais pas identique.
- Les configs : `configs/run1/ppo_agent.yaml` (PPO), `configs/run1/amp_agent.yaml`
  (PPO + D, Table 4 de l'article), `configs/runN/*_env.yaml` (environnement et jeu de clips ; voir `configs/README.md`).
