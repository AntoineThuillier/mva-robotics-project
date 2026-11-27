# Runs

Objectif : les trois tâches AMP (Walk, Run, Locomotion) pour les six séries de `configs/` (voir
`configs/README.md`), chacune avec son évaluation (`eval_policy.txt`) et ses vidéos (`videos/`).
Un run complet fait 300 M échantillons, soit environ 3h55 à 22 000 échantillons/s, plus 10 min
d'évaluation.

## État au 27/11/2026 : tout est fait

Les 18 runs AMP (6 séries × 3 tâches) et le PPO seul sont entraînés jusqu'à 300 M et évalués, avec
leurs vidéos. Résultats : `rl_runs.md`. La file de `runs.sh` a fini le 27/11 vers 08:00.

Pour lancer d'autres runs : écrire la file dans la section `queue` de `runs.sh` (`amp <tâche> <série>`),
activer le venv (`.venv\Scripts\Activate.ps1`, sinon `numpy` est introuvable), puis `bash runs.sh`.
Ne pas modifier `runs.sh` pendant qu'il tourne : bash relit le script en cours d'exécution. Laisser
l'ordinateur sur secteur, sans veille. Pour évaluer un run à la main : `bash runs.sh eval <run>`.

Piste non testée : le plafond de Walk à ~1,3 m/s (séries 2, 5, 6), voir `rl_runs.md`, série 2. Le
départager demanderait plusieurs graines (`--seed`) pour un même Walk.

## Suivi

- Console du run en cours : `Get-Content outputs/rl/<run>.out -Wait -Tail 20` (PowerShell) ou
  `tail -f outputs/rl/<run>.out` (bash).
- Courbes de toutes les séries : `results.ipynb`, section « Séries 2 à 6 ». Le notebook ne fait que
  lire les logs, on peut le relancer pendant un run. Il faut le relancer pour y voir les derniers runs.
- Résultats et validation de chaque série : `rl_runs.md`.
