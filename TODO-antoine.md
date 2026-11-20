# TODO

Regarder code phase 3 : ppo

## Détection de chute cassée quand le nombre d'environnements n'est pas une puissance de 2

**Constat (19 nov. 2026, phase 3).** Avec certains nombres d'environnements, le capteur de contact
avec le sol renvoie une force nulle pour une partie des environnements. Leurs chutes ne sont
alors jamais détectées : le personnage reste allongé au sol jusqu'à la fin des 20 s de l'épisode.
Mesuré avec des actions aléatoires sur 3 s :

| Nombre d'environnements | Chutes détectées |
|---|---|
| 32 | 32/32 |
| 40 | 16/40 |
| 48 | 16/48 |
| 64 | 64/64 |
| 100 | 24/100 |

1, 16, 32, 64 et 4096 fonctionnent (4096 de façon indirecte : les épisodes de test de
l'entraînement durent 0,6 s).

**Contournement actuel.** `build_heading_env` (`amp/heading_env.py`) refuse tout nombre
d'environnements qui n'est pas une puissance de 2. C'est une règle empirique, pas une correction :
la cause n'est pas connue.

**À faire : trouver la vraie cause et corriger la lecture des forces.**
- Où regarder : `reference/MimicKit/mimickit/engines/newton_engine.py`,
  `_build_ground_contact_sensor`, qui lit `newton.sensors.SensorContact(...).net_force`. Le
  tableau est ensuite découpé avec `view(num_envs, -1, 2, 3)`, en supposant que les corps de chaque
  environnement s'y suivent dans l'ordre.
- Hypothèse (non vérifiée) : le capteur ne range pas toujours ses résultats dans cet ordre, si bien
  que le découpage attribue les forces aux mauvais environnements.
- Pour vérifier : avec 40 environnements, comparer l'ordre des corps du capteur (ses attributs
  d'indices ou de noms) à l'ordre attendu par MimicKit, et regarder dans quel environnement
  atterrit la force d'un personnage posé seul au sol.
- Une fois corrigé, l'assertion de `build_heading_env` pourra être retirée.
