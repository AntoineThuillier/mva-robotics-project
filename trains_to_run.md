# Runs à faire

Objectif : les trois tâches AMP (Walk, Run, Locomotion) pour les six séries de `configs/` (voir
`configs/README.md`), chacune avec son évaluation (`eval_policy.txt`) et ses vidéos (`videos/`).
Un run complet fait 300 M échantillons, soit environ 4 à 4,5 h, plus 10 min d'évaluation.

## État au 24/11/2026

| Série | Walk | Run | Locomotion |
|---|---|---|---|
| 1 : référence, `disc_steps` 32 | fait, vidéos | fait, vidéos | fait, vidéos |
| 2 : `disc_steps` 8 | **arrêté à 79 M, à relancer** | fait, *éval. manquante* | **à faire** |
| 3 : régularisation de D | fait, *éval. manquante* | fait, *éval. manquante* | fait, *éval. manquante* |
| 4 : pas de politique 4e-6, pieds alignés | **à faire** | **arrêté à 85 M, à relancer** | **à faire** |
| 5 : entrées constantes masquées | **à faire** | en cours (lancé par l'ancien `runs_rl.sh`) | **à faire** |
| 6 : normaliseur agent + mocap | **à faire** | **à faire** | **à faire** |

PPO seul (`ppo_heading`, série 1) : fait, vidéos.

## File de `runs.sh`, dans l'ordre

`bash runs.sh` fait tout ce qui suit. On peut l'arrêter entre deux runs et retirer les lignes déjà
faites dans la section `queue` du script avant de le relancer.

1. Évaluation et vidéos des runs finis qui n'en ont pas : `amp_run_2`, `amp_run_3`, `amp_walk_3`,
   `amp_locomotion_3`, `amp_run_5`. Compter environ 50 min.
2. Les deux corrections du cou de Run, dans l'ordre :
   1. `amp_run_6`, prioritaire : c'est le test de la seconde correction ;
   2. `amp_walk_5` ;
   3. `amp_locomotion_5` ;
   4. `amp_walk_6` ;
   5. `amp_locomotion_6`.
3. Compléter les séries 2 et 4 : `amp_walk_2`, `amp_locomotion_2`, `amp_run_4`, `amp_walk_4`,
   `amp_locomotion_4`. Pour `amp_walk_2` et `amp_run_4`, le dossier du run arrêté est écrasé. Copie-le
   ailleurs avant si tu veux garder les courbes partielles.

Total pour les points 2 et 3 : 10 runs, soit environ 45 h.

## Après `amp_run_5`

L'ancien script continue avec son ancienne liste. Il échouera tout seul sur `amp_walk_5`, parce que
ses chemins de config n'existent plus, et s'arrêtera. Le `outputs/rl/amp_walk_5.out` d'erreur sera
écrasé plus tard. Ensuite :

```bash
bash runs.sh                     # la file ci-dessus, évaluation de amp_run_5 comprise
```

## Suivi

- Console d'un run : `tail -f outputs/rl/<run>.out`.
- Courbes de toutes les séries : `results.ipynb`, section « Séries 2 à 6 ». Le notebook ne fait que lire
  les logs, on peut le relancer pendant un run.
- Ce qu'on attend sur Run : en séries 5 et 6, r_S ne doit pas s'effondrer vers 0,05 entre 13 et 90 M
  échantillons. En séries 1 à 4, il était à 0,09–0,21 à 50 M.
