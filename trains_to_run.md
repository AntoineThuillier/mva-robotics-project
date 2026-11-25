# Runs à faire

Objectif : les trois tâches AMP (Walk, Run, Locomotion) pour les six séries de `configs/` (voir
`configs/README.md`), chacune avec son évaluation (`eval_policy.txt`) et ses vidéos (`videos/`).
Un run complet fait 300 M échantillons, soit environ 3h55 à 22 000 échantillons/s, plus 10 min
d'évaluation.

## État au 25/11/2026, 09:30

| Série | Walk | Run | Locomotion |
|---|---|---|---|
| 1 : référence, `disc_steps` 32 | fait, évalué | fait, évalué | fait, évalué |
| 2 : `disc_steps` 8 | **arrêté à 79 M, à relancer** | fait, évalué | **à faire** |
| 3 : régularisation de D | fait, évalué | fait, évalué | fait, évalué |
| 4 : pas de politique 4e-6, pieds alignés | **à faire** | **arrêté à 85 M, à relancer** | **à faire** |
| 5 : entrées constantes masquées | fait, évalué | fait, évalué | en cours |
| 6 : normaliseur agent + mocap | **à faire** | fait, évalué | **à faire** |

PPO seul (`ppo_heading`, série 1) : fait, évalué.

## File de `runs.sh`, en cours

`runs.sh` tourne depuis le 24/11 à 22:44. Les évaluations de la file, `amp_run_6` et `amp_walk_5`
sont faites. Il reste, dans l'ordre, avec les heures de fin estimées (évaluation comprise, ±1 h
sur les derniers) :

| Run | Fin estimée | Rôle |
|---|---|---|
| `amp_locomotion_5` | 25/11 ~12:15 | série 5 |
| `amp_walk_6` | 25/11 ~16:20 | série 6. Dira si le plafond de `amp_walk_5` vient des pieds alignés |
| `amp_locomotion_6` | 25/11 ~20:25 | série 6 |
| `amp_walk_2` | 26/11 ~00:30 | compléter la série 2 (écrase le run arrêté) |
| `amp_locomotion_2` | 26/11 ~04:35 | série 2 |
| `amp_run_4` | 26/11 ~08:40 | compléter la série 4 (écrase le run arrêté) |
| `amp_walk_4` | 26/11 ~12:45 | série 4 |
| `amp_locomotion_4` | 26/11 ~16:50 | série 4 |

Le 25/11, entre 08:20 et 09:20, `amp_locomotion_5` a perdu environ 1 h, sans qu'un autre programme
occupe le GPU (mise en veille ou bridage probable). Laisser l'ordinateur sur secteur, sans veille.

Pour arrêter la file : Ctrl+C entre deux runs, puis retirer les lignes déjà faites de la section
`queue` de `runs.sh` avant de le relancer. Pour évaluer un run à la main : `bash runs.sh eval <run>`.

## Suivi

- Console du run en cours : `Get-Content outputs/rl/<run>.out -Wait -Tail 20` (PowerShell) ou
  `tail -f outputs/rl/<run>.out` (bash).
- Courbes de toutes les séries : `results.ipynb`, section « Séries 2 à 6 ». Le notebook ne fait que
  lire les logs, on peut le relancer pendant un run.
- Résultats et validation de chaque série : `rl_runs.md`.
