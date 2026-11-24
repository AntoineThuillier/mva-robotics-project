# Phase 0 : installation et prise en main de MimicKit

Validation décrite dans `CADRAGE.md` (section 4, phase 0). Script : `validation/check_mimickit_install.py`.

**Bilan : environnement en place, entraînement AMP de MimicKit sous Newton conforme à leur log
jusqu'au décollage, débit suffisant. Le modèle pré-entraîné ne marche pas sous Newton.**

## Environnement

| | |
|---|---|
| Date | 15 nov. 2026 |
| Machine | `quetzal` (salle info), NVIDIA RTX A5000 24 Go, GPU partagé avec d'autres utilisateurs |
| Python | 3.11.10 (`~/python311`), venv du projet `.venv/` |
| Paquets | `newton 1.0.0`, `warp-lang 1.12.1`, `mujoco-warp 3.5.0.2`, `torch 2.15.0+cu132` |
| MimicKit | `reference/MimicKit`, commit `fbe1a36` (17 oct. 2026), données dans `data/` |

Les runs de cette phase ont été faits avec un venv installé dans `/tmp`, avec les mêmes
versions, avant le passage au venv du projet.

## Commandes

```bash
.venv/bin/python validation/check_mimickit_install.py pretrained              # rejoue le modèle pré-entraîné
.venv/bin/python validation/check_mimickit_install.py train --minutes 180     # entraînement + comparaison au log
.venv/bin/python validation/check_mimickit_install.py compare outputs/mimickit_install/train/train.out
```

## Résultats

### 1. Modèle pré-entraîné sous Newton : ÉCHEC

`amp_steering_humanoid_model.pt` (entraîné par eux sous Isaac Gym), 256 envs, 512 épisodes.

| | Nous (Newton) | Leur log (Isaac Gym) |
|---|---|---|
| Durée moyenne des épisodes | 80,0 pas (≈ 2,7 s) | 299,5 pas |
| Retour moyen | 37,6 | 222,1 |

Même résultat que sur Mac. La politique entraînée sous Isaac Gym ne se transfère pas à Newton.
Sous Newton, MimicKit démarre les épisodes avec la vitesse de l'origine du bassin là où Newton
attend celle de son centre de masse (écart moyen ≈ 0,1 m/s, `check_heading_env.md`, test 2). Les vitesses
articulaires, elles, sont dans la bonne convention. Cet écart est trop faible pour expliquer à
lui seul l'échec du transfert (non vérifié).

### 2. Entraînement `amp_steering` sous Newton, 3 h : OK jusqu'au décollage

4096 envs, 180 min, arrêt prévu à 76,4 M échantillons (583 itérations).

| Échantillons | Durée des épisodes de test, nous (réf.) | Retour de test, nous (réf.) |
|---|---|---|
| 0,1 M | 17,1 (19,7) | 3,0 (4,0) |
| 26,3 M | 43,2 (27,6) | 8,1 (6,7) |
| 52,6 M | 120,0 (57,0) | 32,2 (14,1) |
| 78,8 M | **282,8 (288,0)** | **110,7 (127,3)** |

La courbe décolle au même moment que la leur, et même un peu plus tôt. À 79 M, le personnage
tient presque tout l'épisode. Le retour d'entraînement suit le leur jusqu'à 65 M (27,1 contre
29,5). À 76 M, il est en retard (39,7 contre 91,4), parce que leur courbe monte très vite à cet
endroit, mais le point de test à 79 M montre qu'il rattrape.

**Limite** : seul le décollage est comparé. Leur log va jusqu'à 2,5 G échantillons
(retour ≈ 220) ; vérifier le plateau demanderait un run de plusieurs jours.

### 3. Débit : OK

| | Échantillons/s | 100 M | 500 M |
|---|---|---|---|
| Nous, run de 3 h | 8 028 | 3,5 h | 17,3 h |
| Nous, run de 20 min | 6 757 | 4,1 h | 20,6 h |
| Leur log (Isaac Gym) | 26 772 | | |

Critère du cadrage (500 M en moins de 24 h) : tenu. Le GPU était partagé, donc le débit
mesuré est un minorant.

## Sorties (non versionnées)

- `outputs/mimickit_install/pretrained/test.out`
- `outputs/mimickit_install/train/` : `model.pt`, `log.txt`, `train.out` (run de 3 h)
- `outputs/mimickit_install/train_3h.report` : rapport de `check_mimickit_install.py`
