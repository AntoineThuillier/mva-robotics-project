# Phase 1 : les données « positives » du discriminateur

Validation décrite dans `CADRAGE.md` (section 4, phase 1). Scripts : `validation/phase1.py`
(tests) et `validation/phase1_view_clip.py` (vidéo d'un clip joué dans Newton).
Code testé : `amp/disc_features.py` (Φ) et `amp/expert.py` (transitions expertes).

**Bilan : 8 tests sur 8 OK. Φ et le tirage des transitions expertes sont validés. Seule la
vitesse de la racine de la mocap doit être convertie dans la convention de Newton (c'est fait
dans `amp/expert.py`).**

> **Correction (19 nov. 2026, après la phase 2).** La première version de cette phase concluait
> qu'il fallait aussi tourner les vitesses des articulations sphériques vers le repère du parent.
> Ce n'est vrai que pour `eval_fk` : le solveur de Newton, qui fait la simulation, utilise le
> repère de l'enfant, comme la mocap (`validation/phase2.md`, test 2). La rotation a été retirée
> de `amp/expert.py`, et le test 1b documente maintenant ce piège. Les résultats ci-dessous du
> test 1b et de la vidéo sont ceux de la version actuelle.

## Environnement

| | |
|---|---|
| Date | 17 nov. 2026, commit `efa1662` |
| Machine | `quetzal`, NVIDIA RTX A5000, `.venv/` du projet (versions dans `phase0.md`) |
| Données | 56 clips du jeu Locomotion (`data/datasets/dataset_humanoid_locomotion.yaml`) |

## Commandes

```bash
.venv/bin/python validation/phase1.py --num_envs 256          # tous les tests
.venv/bin/python validation/phase1.py --tests 3 4             # quelques tests
.venv/bin/python validation/phase1_view_clip.py --clip humanoid_run [--raw]
```

## Résultats (256 états tirés au hasard)

| Test | Critère | Résultat |
|---|---|---|
| 1. Φ(mocap) = Φ(simulateur placé dans le même état) | écart < 1e-3 | OK, 1,8e-4 (mains et pieds) |
| 1b. Vitesses des corps calculées par `eval_fk` vs mocap (piège) | `eval_fk` colle à 1e-2 près seulement si les articulations sont tournées vers le repère du parent | OK, 3,1e-3 rad/s et 1,6e-3 m/s ; convention de l'expert (celle du solveur) : 10,6 rad/s (64 états) |
| 2. Contenu de Φ vs DeepMimic (tailles) et MimicKit (valeurs) | même contenu | OK, détail plus bas |
| 3. Vitesse horizontale du bassin par jeu | Walk ≈ 1 m/s, Run entre 1,5 et 5 m/s | OK, médianes 0,92 et 2,42 m/s |
| 4. Clips et durées vs article, Table 2 | identiques (± 0,5 s) | OK |
| 5. Invariances (rotation autour de z, translation) | écart < 1e-3 | OK, 4e-6 (témoin : bascule de 0,5 rad → 6,45) |
| 6. Pas de NaN | tout fini | OK, 51 986 transitions, avec un avertissement |
| 7. Transitions expertes (`amp/expert.py`) | 4 contrôles | OK |

### Test 1 : écart par bloc de Φ

| Bloc | Écart max |
|---|---|
| hauteur, rotation du bassin | 0 |
| rotules | 7,9e-5 |
| charnières | 2,4e-7 |
| mains et pieds | 1,8e-4 |
| vitesses | 0 (par construction : on relit ce qu'on a écrit, d'où le test 1b) |

L'écart vient du float32 : la cinématique directe de Newton et celle de MimicKit ne font pas
les calculs dans le même ordre.

### Test 2 : Φ comparé à DeepMimic

Référence : `DeepMimicCore/scenes/SceneImitateAMP.cpp` (`RecordAMPObsPose`, `RecordAMPObsVel`).
La tâche Heading active `--enable_amp_obs_local_root true`, donc un repère du cap comme le nôtre.

| Bloc | Nous | DeepMimic | Différence |
|---|---|---|---|
| hauteur du bassin | 1 | 1 | |
| rotation du bassin (6D) | 6 | 6 | DeepMimic range la normale avant la tangente |
| rotations des articulations | 52 | 52 | DeepMimic suit l'ordre des articulations ; nous, rotules puis charnières |
| mains et pieds | 12 | 12 | |
| vitesse du bassin | 3 | 3 | |
| vitesse angulaire du bassin | 3 | 3 | |
| vitesses articulaires | 28 | 36 | DeepMimic stocke 4 nombres par rotule (3 + un 0 de remplissage) |
| **total par état** | **105** | **113** | |

Paire : DeepMimic range `pose(s'), pose(s), vel(s'), vel(s)`, nous `Φ(s), Φ(s')`. Pour le réseau,
toutes ces différences ne sont que des permutations ou des zéros constants.

Valeurs comparées à l'observation de MimicKit (`compute_char_obs`) : tous les blocs communs à
moins de 5e-7. Cette comparaison n'est qu'en partie indépendante, puisque les deux codes
utilisent les mêmes fonctions de `torch_util`.

### Test 3 : vitesse du bassin par jeu de données

| Jeu | Frames | 5 % | Médiane | 95 % |
|---|---|---|---|---|
| Walk | 27 537 | 0,47 | 0,92 | 1,43 |
| Run | 24 505 | 0,65 | 2,42 | 4,36 |

Figure : `outputs/phase1/root_speed_hist.png`. Run a une queue lente (démarrages et virages des
clips `sie`) : la plage « 1,5 à 5 m/s » du cadrage ne vaut que pour le cœur de la distribution.

### Test 4 : clips et durées

| Jeu | Clips | Durée | Article (Table 2) |
|---|---|---|---|
| Walk | 9 | 229,4 s | 9 clips, 229,6 s |
| Run | 47 | 204,4 s | 47 clips, 204,4 s |

### Test 7 : transitions expertes

| Contrôle | Résultat |
|---|---|
| (a) forme et NaN | (256, 210), aucun NaN |
| (b) paires dans un seul clip | t ≥ 0,012 s, marge ≥ 0,005 s avant la fin du clip |
| (c) clips tirés selon les poids du yaml (100 000 tirages) | écart max 2,3 écarts-types (seuil 4) |
| (d) Φ(s') placé dans Newton et recalculé | écart 1,4e-4 ; avec les vitesses brutes, 15,7 : la conversion est bien appliquée |

### Vidéo d'un clip joué dans Newton

`humanoid_run`, 24 frames : positions Newton = mocap à 5e-7 m près. Vitesses des mains et des pieds :
écart de 3e-6 m/s avec conversion, jusqu'à 5,6 m/s sans. Vidéos : `outputs/phase1/clip_*.mp4`.
Les flèches de Newton sont calculées par `eval_fk` : les vitesses y sont donc écrites dans la
convention de `eval_fk` (articulations dans le repère du parent), pas dans celle du solveur.

## Ce que la phase a révélé

1. **La vitesse du bassin n'est pas écrite dans la même convention dans la mocap et dans Newton.**
   MimicKit donne la vitesse de l'origine, Newton celle du centre de masse (≈ 11 cm au-dessus).
   Conversion : v_com = v_origine + ω × r.

   Pour les rotules, MimicKit (`MotionLib`) exprime la vitesse dans le repère de l'enfant
   (q0⁻¹·q1). Le solveur de Newton aussi (`validation/phase2.md`, test 2) : rien à convertir.
   Seul `eval_fk` les lit dans le repère du parent, d'où la conclusion erronée de la première
   version de cette phase (voir la correction en tête).

   La couche Newton de MimicKit (`newton_engine.py`) recopie les nombres sans les convertir.
   Sans conversion, le discriminateur verrait une différence systématique sur les vitesses
   entre l'expert et l'agent. La conversion est faite dans `amp/expert.py`
   (`mocap_velocities_to_newton`) et au démarrage des épisodes (`amp/heading_env.py`, phase 2).

2. **`slerp` de MimicKit** (`util/torch_util.py`) : si deux frames diffèrent de moins de 0,002 rad,
   il renvoie la moyenne fixe des deux, et la pose interpolée reste figée pendant la frame.
   C'est fréquent à 120 fps. Effet négligeable sur les poses ; ça interdit de vérifier les
   vitesses par différence finie (d'où le calcul exact du test 1b).

3. **`body_vel` de Newton est faux juste après `eval_fk`** (`newton/_src/sim/articulation.py`) :
   la vitesse du parent est propagée comme celle de son centre de masse, mais c'est celle de
   l'origine qui est stockée pour l'enfant. L'erreur grandit le long du squelette, jusqu'à
   ≈ 2 m/s sur les mains et les pieds. Ne pas lire `get_body_vel` après un placement d'état ;
   `newton_body_origin_vel` (dans `phase1.py`) reconstruit les vraies vitesses.

4. **NaN sur le cou de `humanoid_walk.pkl`** (colonnes 9 à 11, 125 frames) : au chargement,
   MimicKit les remplace en silence par une rotation nulle. Dans ce clip, le cou reste figé droit.

## Pas encore vérifié

- Les vitesses linéaires des corps après un vrai pas de simulation (on n'a testé que l'état placé).
- L'effet de la convention des vitesses sur l'AMP de MimicKit lui-même (référence de contrôle).
