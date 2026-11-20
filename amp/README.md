# `amp/` : notre code de la réimplémentation d'AMP

Ce dossier contient ce que nous écrivons nous-mêmes. Le reste (simulateur Newton, personnage,
contrôle PD, chargement de la mocap, PPO) vient de MimicKit, cloné dans `reference/MimicKit`.
Le projet avance par phases (`CADRAGE.md`, section 4). Chacune a son script de validation
`validation/phaseN.py` et son bilan `validation/phaseN.md`.

## Phase 1 : les données « positives » du discriminateur

Le discriminateur apprend à distinguer les transitions (s, s') de la mocap de celles du
personnage simulé. Cette phase fournit ce qu'il voit, Φ(s), et les transitions de la mocap.

- `disc_features.py` : calcule Φ(s), un vecteur de 105 nombres (hauteur et rotation de la racine,
  rotations des articulations, positions des mains et des pieds, vitesses), exprimé dans le
  repère du cap. `phi_paire(s, s')` donne l'entrée du discriminateur.
- `expert.py` : tire des transitions de la mocap (`ExpertTransitions`) et convertit leurs
  vitesses vers la convention de Newton (seule la vitesse de la racine change, voir
  `validation/phase2.md`, test 2).
- Validation : `validation/phase1.py`, `validation/phase1_view_clip.py`, `validation/phase1.md`.

## Phase 2 : l'environnement Target Heading

Le monde dans lequel la politique agit : le personnage doit aller dans une direction cible à une
vitesse cible (article, section 7). L'environnement donne le goal, calcule la récompense de
tâche r_G (éq. 11) et décide de la fin des épisodes.

- `heading_env.py` : la classe `TaskHeadingEnv`, qui hérite de l'environnement de MimicKit et ne
  change que le goal, son calendrier, la récompense, la durée des épisodes et la vitesse de la
  racine au démarrage. `build_heading_env(...)` la construit.
- `../configs/amp_heading_humanoid_env.yaml` : sa configuration.
- `../configs/newton_test_nopd_240hz.yaml` : une config du simulateur pour un seul test (sans PD).
- Validation : `validation/phase2.py`, `validation/phase2_videos.py` (vidéos pour le rapport),
  `validation/phase2.md`. Les environnements des tests sont
  construits par `validation/load_envs.py`, une seule fois par processus.

## Phase 3 : PPO seul (récompense de tâche uniquement)

On entraîne une politique avec PPO (celui de MimicKit) sur l'environnement Target Heading, avec
la seule récompense de tâche r_G, sans discriminateur. Ça valide la boucle d'apprentissage avant
la phase 4, et donne l'équivalent de la baseline « No Data » de l'article (Fig. 5), à ceci près
que les épisodes démarrent depuis des poses de la mocap.

- `run.py` : lance l'entraînement ou le test avec le `run.py` de MimicKit, après y avoir
  enregistré notre environnement (`task_heading`).
- `../configs/ppo_heading_humanoid_agent.yaml` : l'agent PPO (réglages de l'AMP de MimicKit,
  sans le discriminateur).
- `../configs/ppo_heading_humanoid_args.txt` : les arguments du run.
- Validation : `validation/phase3.py` (sur un dossier de run), `validation/phase3_videos.py`,
  `validation/phase3.md`.

## Utilisation

Le code de ce dossier importe les modules de MimicKit. Il faut donc avoir `reference/MimicKit/mimickit`
et la racine du projet dans le `sys.path`, et lancer depuis la racine de MimicKit, comme le font
les scripts de `validation/` :

```python
os.chdir(MIMICKIT)
sys.path[:0] = [str(MIMICKIT / "mimickit"), str(PROJECT)]
from amp.heading_env import build_heading_env
```
