# Runs à faire

Objectif : les trois tâches AMP (Walk, Run, Locomotion) pour les six séries de `configs/` (voir
`configs/README.md`), chacune avec son évaluation (`eval_policy.txt`) et ses vidéos (`videos/`).
Un run complet fait 300 M échantillons, soit environ 3h55 à 22 000 échantillons/s, plus 10 min
d'évaluation.

## État au 25/11/2026, 23:30

| Série | Walk | Run | Locomotion |
|---|---|---|---|
| 1 : référence, `disc_steps` 32 | fait, évalué | fait, évalué | fait, évalué |
| 2 : `disc_steps` 8 | en cours (relancé) | fait, évalué | **à faire** |
| 3 : régularisation de D | fait, évalué | fait, évalué | fait, évalué |
| 4 : pas de politique 4e-6, pieds alignés | **à faire** | **arrêté à 85 M, à relancer** | **à faire** |
| 5 : entrées constantes masquées | fait, évalué | fait, évalué | fait, évalué |
| 6 : normaliseur agent + mocap | fait, évalué | fait, évalué | fait, évalué |

PPO seul (`ppo_heading`, série 1) : fait, évalué.

## File de `runs.sh`, en cours

`runs.sh` a été relancé le 25/11 à 18:38 avec la file restante. `amp_locomotion_6` est fini et
évalué (22:41). Il reste, avec les heures de fin estimées (évaluation comprise, ±1 h sur les derniers) :

| Run | Fin estimée | Rôle |
|---|---|---|
| `amp_walk_2` | 26/11 ~02:35 (en cours) | compléter la série 2 (écrase le run arrêté) |
| `amp_locomotion_2` | 26/11 ~06:40 | série 2 |
| `amp_run_4` | 26/11 ~10:45 | compléter la série 4 (écrase le run arrêté) |
| `amp_walk_4` | 26/11 ~14:50 | série 4. Troisième Walk avec pieds alignés : plafonne-t-il aussi ? |
| `amp_locomotion_4` | 26/11 ~18:55 | série 4 |

Ne pas modifier `runs.sh` pendant qu'il tourne : bash relit le script en cours d'exécution.
Laisser l'ordinateur sur secteur, sans veille (le 25/11, `amp_locomotion_5` a perdu environ 1 h).

Pour arrêter la file : Ctrl+C entre deux runs, puis retirer les lignes déjà faites de la section
`queue` de `runs.sh` avant de le relancer. Pour évaluer un run à la main : `bash runs.sh eval <run>`.

## Suivi

- Console du run en cours : `Get-Content outputs/rl/<run>.out -Wait -Tail 20` (PowerShell) ou
  `tail -f outputs/rl/<run>.out` (bash).
- Courbes de toutes les séries : `results.ipynb`, section « Séries 2 à 6 ». Le notebook ne fait que
  lire les logs, on peut le relancer pendant un run.
- Résultats et validation de chaque série : `rl_runs.md`.
