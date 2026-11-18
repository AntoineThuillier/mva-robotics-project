# Phase 2 : l'environnement Target Heading

Validation décrite dans `CADRAGE.md` (section 4, phase 2). Script : `validation/phase2.py`.
Code testé : `amp/heading_env.py` et sa config `configs/amp_heading_humanoid_env.yaml`.

**Bilan : 6 tests sur 6 OK. L'environnement suit l'article (goal, calendrier de DeepMimic,
éq. 11, épisodes de 20 s). Deux résultats changent le plan :**
1. **Seule la vitesse de la racine doit être convertie** au démarrage des épisodes. Les vitesses
   des articulations sphériques sont déjà dans la convention du simulateur. Cela contredit en
   partie la phase 1 : voir « Ce que la phase a révélé ».
2. **Le suivi de la mocap sans politique ne tient pas « quelques secondes »** comme prévu par le
   cadrage (médiane 0,8 s). Le critère a été remplacé.

## Environnement

| | |
|---|---|
| Date | 18 nov. 2026 |
| Machine | PC personnel, NVIDIA RTX 3070 (8 Go), Windows 11 |
| Paquets | Python 3.11, `newton 1.0.0`, `warp-lang 1.12.1`, `mujoco 3.5.0`, `mujoco-warp 3.5.0.2`, `torch 2.14.1+cu130` |
| MimicKit | `reference/MimicKit`, commit `fbe1a36` |

`mujoco` doit être en 3.5.0 : la 4.x casse `mujoco-warp` (corrigé dans `requirements.txt`).

## Commandes

```bash
.venv/Scripts/python validation/phase2.py                # tous les tests, 64 envs, environ 1 min
.venv/Scripts/python validation/phase2.py --tests 3 4
.venv/Scripts/python validation/phase2_videos.py         # vidéos des tests 1 et 2, environ 1 min
```

Les environnements viennent de `validation/load_envs.py`, qui ne les construit qu'une fois par
processus. Dans un notebook ou avec `python -i`, on peut donc lancer les tests l'un après l'autre
(`phase2.test_random_policy()`, puis `phase2.test_reset_and_tracking()`…) sans reconstruire
l'environnement.

## Ce que fait l'environnement

`TaskHeadingEnv` hérite de `TaskSteeringEnv` de MimicKit et ne change que ceci :

- **goal** g = (direction cible vue depuis le personnage (2), v*) : 3 nombres. MimicKit ajoute
  une direction de regard, qui est retirée. DeepMimic écrit (cos Δ, −sin Δ, v*) dans son repère
  y vertical ; c'est la même grandeur que (cos Δ, sin Δ) dans le repère z vertical de MimicKit ;
- **calendrier du goal**, celui de DeepMimic (`SceneHeadingAMP.cpp` et
  `args/train_amp_heading_humanoid3d_locomotion_args.txt`). Toutes les 0,2 à 0,5 s, le cap dérive
  de N(0 ; 0,15 rad). Avec une probabilité de 0,01, c'est un virage brusque U(−π, π) ; avec une
  probabilité de 0,02, v* est retirée dans U(1, 5) m/s. Les probabilités s'appliquent à chaque
  mise à jour, pas à chaque pas ;
- **récompense r_G** : éq. 11 dans la version de DeepMimic. La vitesse utilisée est la vitesse
  moyenne du centre de masse sur le pas de 1/30 s, projetée sur la direction cible ; r_G vaut 0
  si le personnage recule. Le goal utilisé est celui en vigueur pendant le pas, même si une mise
  à jour tombe juste après ;
- **démarrage des épisodes** : la vitesse de la racine est convertie de l'origine du bassin
  (mocap) vers son centre de masse (Newton) ;
- **épisodes de 20 s**, v* dans [1, 5] m/s.

## Résultats (64 envs)

| Test | Critère | Résultat |
|---|---|---|
| 1. Politique aléatoire | tous tombent en moins de 3 s, médiane < 1 s | OK, médiane 0,50 s (0,27 à 0,73 s) |
| 2. Démarrage depuis la mocap et conventions de Newton | voir plus bas | OK |
| 3. Récompense r_G | éq. 11 à la main, à 1e-5 | OK, écart max 3,6e-7 |
| 4. Calendrier du goal | DeepMimic, à 4 écarts-types | OK |
| 5. Observation g | formule, à 1e-5 | OK, écart max 4,5e-7 |
| 6. Durée des épisodes | fin à 20 s, drapeau TIME | OK, fin au pas 600 pour tous |

### Test 2 : démarrage depuis la mocap

**(a) Vitesses écrites par le reset** : celles attendues, à 0 près. La conversion change la
vitesse de la racine de 0,10 m/s en moyenne.

**(b) Conventions du simulateur**, mesurées sans contrôle PD, sur un seul pas de 1/240 s, avec le
personnage 1,5 m au-dessus du sol (aucun contact). On écrit des vitesses aléatoires et on regarde
le mouvement produit. Erreur moyenne selon la convention supposée :

| Grandeur | Convention qui colle | Erreur | Autre convention | Erreur |
|---|---|---|---|---|
| Articulations sphériques, intégration | repère de l'**enfant** | 0,28 rad/s | repère du parent | 1,70 rad/s |
| Articulations sphériques, relecture | repère de l'**enfant** | 0,28 rad/s | repère du parent | 1,70 rad/s |
| Vitesse linéaire de la racine | **centre de masse** | 0,03 m/s | origine | 0,46 m/s |
| Vitesse angulaire de la racine | repère **monde** | 0,22 rad/s | repère de la racine | 4,94 rad/s |

L'erreur résiduelle vient de la gravité et du couplage entre les corps pendant le pas.

**(c) Suivi de la mocap** (la pose suivante du clip devient la cible PD). L'erreur de position
des corps après un pas est de 1,4 cm (3,1 cm après 2 pas, 5,0 cm après 3). Durée des épisodes,
sans critère :

| Politique | Médiane | Maximum |
|---|---|---|
| Suivi de la mocap | 0,83 s | 1,97 s |
| Actions nulles | 0,70 s | 4,00 s |
| Actions aléatoires | 0,43 s | 0,83 s |

**Écart au cadrage** : le cadrage demandait que le personnage suive le clip « pendant au moins
quelques secondes ». Ce n'est pas atteignable sans politique apprise : le PD seul accumule un
retard de 0,1 à 0,25 rad par articulation en quelques pas, sans aucun rattrapage de l'équilibre,
et le personnage tombe (souvent une main au sol dans les courses). C'est justement ce que
l'apprentissage doit corriger. Le critère a donc été remplacé par l'erreur après un pas, qui
vérifie que les cibles PD sont bien lues. La vraie preuve que l'environnement permet de marcher
viendra des phases 3 et 4.

### Test 3 : récompense r_G

| Contrôle | Résultat |
|---|---|
| (a) centre de masse : corps de Newton vs modèle cinématique dans le même état | écart 9,5e-7 m |
| (b) éq. 11 à la main sur 4 000 transitions de mocap (alignée, 60°, 180°, v* + 2 m/s) | écart max 3,6e-7 |
| (c) cible = direction du regard, v* = vitesse du clip | r_G médian 1,000, 5e centile 0,999 |
| (d) récompense de l'env vs recalculée pas à pas, dont 128 pas avec une mise à jour du goal | écart 0 |

### Test 4 : calendrier du goal (64 envs × 300 s, 52 334 mises à jour)

| Grandeur | Mesuré | Attendu |
|---|---|---|
| Intervalle entre mises à jour | moyenne 0,367 s, de 0,200 à 0,500 s | 0,35 s + un demi-pas = 0,367 s |
| Dérive du cap | écart-type 0,1507 rad | 0,15 rad |
| Virages brusques de plus de 0,75 rad | 0,0078 par mise à jour (+0,6 σ) | 0,0076 |
| Changements de vitesse | 0,0198 par mise à jour (−0,4 σ) | 0,02 |

## Ce que la phase a révélé

1. **La conversion des vitesses des articulations sphériques de la phase 1 ne correspond pas au
   simulateur.** La phase 1 a déterminé la convention de Newton avec `eval_fk`, qui calcule les
   vitesses des corps à partir de `joint_qd` en les supposant dans le repère du parent. Mais le
   solveur qui fait la simulation (MuJoCo dans Newton) intègre ces mêmes vitesses dans le repère
   de l'**enfant**, et les relit dans ce repère. C'est la convention de la mocap. Newton n'est donc
   pas cohérent avec lui-même : c'est le même genre de défaut que `body_vel` après `eval_fk`,
   déjà signalé dans `phase1.md`.

   **Conséquences :**
   - au démarrage des épisodes, seule la vitesse de la racine doit être convertie : c'est ce que
     fait `amp/heading_env.py` ;
   - **`amp/expert.py` tourne encore les vitesses des articulations sphériques**
     (`mocap_velocities_to_newton`, point (a)). Or les états de l'agent, relus dans Newton, ont ces
     vitesses dans le repère de l'enfant. Les transitions expertes et celles de l'agent n'auraient
     donc pas la même convention, ce que la phase 1 voulait justement éviter. **À corriger avant la
     phase 4** : retirer le point (a) et garder le point (b) (racine). Les tests 1b et 7 de la
     phase 1 sont à revoir en conséquence, puisqu'ils s'appuient sur `eval_fk`.

2. **Le suivi de la mocap sans politique tombe en moins d'une seconde** (voir test 2).

## Vidéos (pour le rapport)

`validation/phase2_videos.py` produit trois vidéos de 3 s, ralenties ×0,5, dans `outputs/phase2/`
(non versionné). Le personnage simulé est en bleu, le clip suivi en gris, la direction cible en
vert (longueur proportionnelle à v*), la vitesse du centre de masse en orange ; v*, la vitesse
dans la direction cible et r_G sont affichés à chaque pas. Pour le suivi, le goal part aligné
avec le clip (direction du regard, v* = vitesse du clip).

| Vidéo | Politique | v* au départ | Fin de l'épisode |
|---|---|---|---|
| `random.mp4` | actions aléatoires | 4,15 m/s (goal aléatoire) | chute à 0,43 s |
| `track_walk.mp4` | suivi de `long/humanoid_long_walk0` à partir de 10 s | 1,00 m/s | chute à 1,10 s |
| `track_run.mp4` | suivi de `humanoid_run` à partir de 0,5 s | 3,57 m/s | chute à 0,53 s |

## Pas encore vérifié

- La durée de vie d'un personnage contrôlé par une vraie politique : ce sera la phase 3.
