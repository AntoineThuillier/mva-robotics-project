# Cadrage du projet : réimplémentation d'AMP sur Target Heading

Projet MVA « Robotics » : comprendre puis réimplémenter *AMP: Adversarial Motion
Priors for Stylized Physics-Based Character Control* (Peng et al., SIGGRAPH 2021).

- **Poster** : 3 décembre 2026 (premiers résultats).
- **Rapport** (5 à 8 pages) : 18 décembre 2026.

## 1. Ce qu'on fait

On écrit **notre propre version de l'apport de l'article** (les features Φ, le
discriminateur, la récompense de style, la tâche Target Heading) **à l'intérieur de
[MimicKit](https://github.com/xbpeng/MimicKit)**, dont on réutilise l'infrastructure :
moteur physique (Newton), personnage, commande PD, chargement de la mocap, PPO.

On travaille sur une seule tâche : **Target Heading**, avec trois jeux de mocap
(Walk, Run, Locomotion), comme la Table 1 et la Fig. 4 de l'article.

## 2. Les trois références et leur rôle

| Référence | Ce que c'est | Rôle pour nous |
|---|---|---|
| **L'article** (`project-article.pdf`) | la méthode et les résultats publiés | **la référence de validation finale** : on compare nos résultats à ses tableaux et figures |
| **Le code de DeepMimic** ([xbpeng/DeepMimic](https://github.com/xbpeng/DeepMimic), copie locale dans `reference/`, non versionnée) | le code d'origine de l'article (C++/Bullet, TensorFlow) | combler ce que l'article ne précise pas (calendrier du goal, gains PD, détails de la récompense) |
| **L'AMP de MimicKit** | une réimplémentation récente par le même auteur | **une référence de contrôle intermédiaire** : même moteur et même environnement que nous, donc un écart entre leurs courbes et les nôtres signale un bug dans notre code |

Règle quand ces sources divergent : **on suit l'article**. On utilise DeepMimic pour ce
que l'article ne dit pas, et on note chaque écart en section 5.

## 3. Les données

Les 56 clips du jeu Locomotion de l'article (Table 2), au format MimicKit, dans
`data/motions/humanoid/` (non versionné, voir le README pour le téléchargement).

| Jeu | Fichiers | Clips | Durée chez nous | Durée dans l'article |
|---|---|---|---|---|
| Walk | `humanoid_walk`, `long/humanoid_long_walk*` | 9 | 229,4 s | 229,6 s |
| Run | `humanoid_run`, `sie/humanoid_sie_run_*` | 47 | 204,4 s | 204,4 s |
| Locomotion | Walk + Run | 56 | 433,8 s | 434,1 s |

Défauts connus : NaN sur le cou dans `humanoid_walk.pkl`, et pieds 2 à 6 cm sous le sol
dans la plupart des clips.

**Rappel sur le rôle des données** : la politique n'est **jamais** entraînée directement
sur la mocap. La mocap sert seulement à deux choses : les exemples « réels » du
discriminateur, et les états initiaux des épisodes (RSI). La politique apprend sur ses
propres essais dans le simulateur.

## 4. Les phases

Chaque phase se termine par une validation. On ne passe à la suivante que si elle est réussie.

### Phase 0 : installation et prise en main de MimicKit (16-17 nov.)

**Objectif** : faire tourner MimicKit et comprendre son architecture.

- Installer MimicKit et Newton sur le Mac (`--devices cpu`, à confirmer) pour développer.
- Choisir la machine avec GPU NVIDIA pour les entraînements longs (Narval, cluster de
  Polytechnique ou Colab) : le Mac ne suffira pas pour des centaines de millions d'échantillons.
- Lire le chemin d'un pas de simulation : `run.py`, puis `char_env.py` /
  `task_steering_env.py`, puis `engines/newton_engine.py`, `amp_agent.py` et `ppo_agent.py`.
- Lister les classes qu'on garde et celles qu'on remplace.

**Validation**

| Test | Comparé à | Critère |
|---|---|---|
| Rejouer le modèle pré-entraîné `amp_steering_humanoid_model.pt` | l'article, Fig. 3 et vidéo | le personnage suit la direction demandée avec une démarche naturelle |
| Lancer quelques minutes l'entraînement `amp_steering` de MimicKit | leur log `amp_steering_humanoid_log.txt` (fourni dans l'archive) | le retour de test augmente comme dans leur log au même nombre d'échantillons |
| Mesurer le débit (pas de politique par seconde) | — | assez rapide pour environ 500 M d'échantillons en moins d'une journée |

**Ce qu'on obtient** : un schéma de l'architecture MimicKit, et la référence de contrôle
pour toute la suite.

### Phase 1 : les données « positives » du discriminateur (17-19 nov.)

**Objectif** : écrire notre Φ(s) et le tirage des transitions de mocap.

- On réutilise `motion_lib.py` de MimicKit (chargement, interpolation, vitesses).
- On écrit `compute_amp_features(state)`. Le contenu suit l'article (section 5.3) et le code de DeepMimic :
  - la hauteur de la racine et sa rotation en 6D (normale-tangente) ;
  - les rotations des articulations : 6D pour les sphériques, l'angle pour les révolutions ;
  - les positions des mains et des pieds par rapport à la racine ;
  - les vitesses linéaire et angulaire de la racine, et les vitesses articulaires.

  Le tout est exprimé dans le **repère du cap** de l'état t+1, appliqué aux deux états.
- On écrit `sample_expert_transitions(n)` : on tire un clip et un instant continu au
  hasard, puis on calcule la paire (Φ(t − 1/30 s), Φ(t)) par interpolation.
- Le discriminateur voit **2 états**, comme dans l'article (MimicKit en utilise 10).

**Validation**

| Test | Comparé à | Critère |
|---|---|---|
| Φ calculé sur la mocap, puis le simulateur placé dans le même état et Φ recalculé | notre propre calcul | écart nul (précision machine), **vitesses comprises** |
| Dimension de Φ et ordre des blocs | le code de DeepMimic (`SceneImitateAMP.cpp`, `BuildAMPObs`) | même contenu, bloc par bloc |
| Histogrammes de vitesse du bassin par jeu de données | l'article, Fig. 4 et section 8.2 | Walk autour de 1 m/s, Run de 1,5 à 5 m/s : ce sont les plages qui expliquent pourquoi la politique Walk n'atteint pas les vitesses élevées dans la Fig. 4 |
| Durée et nombre de clips | l'article, Table 2 | identiques (déjà vérifié, voir section 3) |

**Ce qu'on obtient** : un module Φ testé, et les figures de distribution des données pour le rapport.

### Phase 2 : l'environnement Target Heading (19-21 nov.)

**Objectif** : un environnement fidèle à l'article, testé avec des politiques factices.

- On réutilise la physique, le PD, le personnage, le RSI et la détection de chute de MimicKit.
- On écrit, ou on adapte depuis `task_steering_env.py` :
  - **le goal** g = (cos Δ, −sin Δ, v*), où Δ est l'angle entre la direction cible et le cap du personnage ;
  - **le calendrier du goal**, pris dans DeepMimic : v* uniforme dans [1, 5] m/s. Toutes les 0,2 à 0,5 s, la direction dérive de N(0 ; 0,15 rad). Avec une probabilité de 1 %, c'est un virage brusque uniforme dans [−π, π]. Avec une probabilité de 2 %, v* est retirée ;
  - **la récompense de tâche** r_G = exp(−0,25·(v* − d*·v̄_com)²) (éq. 11), où v̄_com est la vitesse moyenne du centre de masse sur le pas de 1/30 s. Elle vaut 0 si le personnage recule. Il n'y a **pas** le terme d'orientation du regard de MimicKit ;
  - des épisodes de 20 s, et l'early termination (contact avec le sol d'une autre partie du corps que les pieds).

**Validation, avec des politiques factices**

| Test | Comparé à | Critère |
|---|---|---|
| Politique aléatoire | l'article, section 6.3 (early termination) | le personnage tombe et l'épisode s'arrête en moins d'une seconde |
| Suivi de la mocap : la pose suivante du clip devient la cible PD | la mocap elle-même | le personnage suit le clip pendant au moins quelques secondes, avec une erreur de pose (éq. 10) faible au début |
| Lecture cinématique d'un clip de course aligné sur g | l'éq. 11 calculée à la main | r_G proche de 1 si la direction et la vitesse correspondent ; proche de 0 si on désaligne de 90° ou si v* est loin de la vitesse du clip |
| Statistiques du goal sur de longs épisodes | le code de DeepMimic (`SceneHeadingAMP.cpp`) | même fréquence de virages et de changements de vitesse |

**Ce qu'on obtient** : un environnement validé, et des GIF des tests.

### Phase 3 : PPO seul, avec r = r_G (21-24 nov.)

**Objectif** : valider toute la boucle de renforcement **avant** d'ajouter le discriminateur.

- On réutilise `ppo_agent.py` de MimicKit, avec les réglages de l'article (Table 4) :
  - π et V : MLP 1024-512 ReLU, π gaussienne de covariance fixe ;
  - γ = 0,99 (tâche), GAE(λ = 0,95) et TD(λ = 0,95) ;
  - 4 096 échantillons par itération, minibatches de 256 ;
  - clip de PPO à 0,2 : la Table 4 indique 0,02, sans doute une coquille, puisque le code de DeepMimic utilise 0,2.

**Validation**

| Test | Comparé à | Critère |
|---|---|---|
| Comportement obtenu | l'article, Fig. 5, baseline « No Data » | le personnage avance dans la direction demandée, avec une démarche peu naturelle |
| Retour de tâche normalisé | l'article, Fig. 5 | du même ordre que leur courbe « No Data » ; la tâche est remplie sans le style |
| Durée des épisodes | — | elle augmente : le personnage tombe de moins en moins |

**Ce qu'on obtient** : notre première courbe d'apprentissage et la baseline « No Data ».

### Phase 4 : AMP (24 nov. → 1er déc.)

**Objectif** : notre discriminateur et la récompense de style.

- On écrit notre agent AMP, qui hérite de PPO :
  - D : MLP 1024-512, en entrée (Φ(s), Φ(s′)) normalisés ;
  - la perte LSGAN : (D(réel) − 1)² + (D(faux) + 1)² (éq. 8) ;
  - le gradient penalty (w_gp/2)·‖∇_φ D‖² **sur la mocap uniquement**, avec w_gp = 10 ;
  - la récompense de style r_S = max(0, 1 − 0,25·(D − 1)²) (éq. 7) ;
  - la récompense totale r = 0,5·r_G + 0,5·r_S (éq. 4) ;
  - un replay buffer de 10⁵ transitions de la politique, et des minibatches de 256 transitions réelles et 256 générées ;
  - le pas d'apprentissage de D à 10⁻⁵.
- L'ordre des entraînements : Walk, puis Run, puis Locomotion.

**Validation**

| Test | Comparé à | Critère |
|---|---|---|
| Courbes d'apprentissage sur le même environnement | **l'AMP de MimicKit** (contrôle intermédiaire) | même ordre de grandeur de retour au même nombre d'échantillons ; un écart important signale un bug chez nous |
| Diagnostics de D : précision sur le réel et sur le généré, r_S moyen, valeur du gradient penalty | — | D ne gagne pas totalement (précision qui ne colle pas à 100 %) et ne s'effondre pas (pas bloquée à 50 % dès le début) |
| Comportement de la politique Locomotion | l'article, section 8.2 et Fig. 3 | marche vers 1 m/s, jog vers 2,5 m/s, course vers 4,5 m/s ; le personnage s'incline dans les virages |
| Démarche | les clips de mocap rendus dans `notebooks/01_visualisation_mocap.ipynb` | visuellement proche de la mocap, avec des pellicules façon Fig. 3 |

**Ce qu'on obtient** : trois politiques (Walk, Run, Locomotion) et le contenu du poster.

### Phase 5 : résultats, analyse et extensions (2 → 18 déc.)

**Validation finale : comparaison à l'article**

| Résultat | Référence dans l'article | Valeurs de l'article |
|---|---|---|
| Retour de tâche normalisé, Target Heading, sur 3 graines × 32 épisodes | Table 1 | Locomotion **0,90 ± 0,01**, Walk **0,46 ± 0,01**, Run **0,63 ± 0,01** |
| Vitesse obtenue en fonction de la vitesse cible | Fig. 4 (droite) | Locomotion suit toute la plage [1, 5] m/s ; Walk plafonne aux faibles vitesses ; Run ne sait pas aller lentement |
| Courbes d'apprentissage des trois jeux | Fig. 4 (gauche) | Locomotion au-dessus de Run, lui-même au-dessus de Walk |
| Changements d'allure selon v* | section 8.2 | marche, puis jog, puis course quand v* augmente |
| Comparaison avec PPO seul | Fig. 5 | retour de tâche comparable, mais un style nettement plus naturel avec AMP |

On compare aussi les trois versions : notre AMP, l'AMP de MimicKit et PPO seul.

**Extensions, selon le temps restant**

1. **Le poids réel du style** : 0,5 dans l'article, mais 1,0 en pratique dans le code de
   DeepMimic (`RewardScale` = 2).
2. **L'historique du discriminateur** : 2 états (article) ou 10 (MimicKit).
3. **Les ablations de l'article** (sans gradient penalty, sans features de vitesse), mais
   sur la tâche Target Heading, alors que l'article ne les montre qu'en imitation d'un
   seul clip (Fig. 8).
4. **Une métrique de style chiffrée**, absente de l'article pour les tâches : glissement
   des pieds, ou distance au clip de mocap le plus proche.

**Ce qu'on obtient** : le rapport.

## 5. Écarts identifiés entre l'article, DeepMimic et MimicKit

| Élément | Article | Code DeepMimic | MimicKit | Notre choix |
|---|---|---|---|---|
| Moteur, fréquence de la physique | Bullet, 1,2 kHz | Bullet, 60 × 10 × 2 = 1,2 kHz, PD implicite | Newton 240 Hz, Isaac Gym 120 Hz | Newton (MimicKit) |
| États vus par D | 2 | 2 | 10 | **2** |
| Récompense de Target Heading | éq. 11 | éq. 11 avec la vitesse moyenne sur le pas, et 0 si le personnage recule | éq. 11 + terme d'orientation du regard | **éq. 11, version DeepMimic** |
| Calendrier du goal | non précisé | dérive toutes les 0,2 à 0,5 s, virages brusques et changements de vitesse rares | nouvelle direction toutes les 4 à 7 s | **DeepMimic** |
| Plage de v* | [1, 5] m/s | [1, 5] m/s | [0,5 ; 5] m/s | **[1, 5] m/s** |
| Poids du style | w_S = 0,5 | 0,5 × `RewardScale` 2 | à vérifier en phase 0 | **0,5** (et 1,0 en extension) |
| Clip de PPO | 0,02 (Table 4) | 0,2 | à vérifier en phase 0 | **0,2** |
| Régularisation de D | gradient penalty | GP + weight decay 5·10⁻⁴ + pénalité 0,05 sur la dernière couche | à vérifier en phase 0 | **GP** (le reste si D est instable) |
| Réseaux de π et V | MLP 1024-512 | MLP 1024-512 « gated » par le goal | à vérifier en phase 0 | **MLP 1024-512** |
| Exploration | σ fixe | σ = 0,05 ; action bruitée avec une probabilité qui décroît de 1 à 0,2 | à vérifier en phase 0 | **σ fixe** |
| Durée d'un épisode | non précisée | 20 s | 10 s | **20 s** |

## 6. Risques

- **L'installation de Newton** (phase 0) : si elle échoue, on se replie sur notre pile
  MuJoCo sur CPU et PyTorch (un benchmark de cette pile existe dans l'historique git,
  commit `558e6fb`, `scripts/sim_benchmark.py`).
- **Un personnage instable** (phase 2) : gains PD, fréquence de la physique, pieds qui
  s'enfoncent dans le sol au démarrage.
- **Le budget de calcul** : l'article utilise 100 à 300 M d'échantillons par politique,
  et le log de MimicKit environ 630 M pour atteindre 60 % du retour maximal.
- **Un AMP qui n'apprend pas** (phase 4) : l'AMP de MimicKit sert alors de filet de
  sécurité pour le poster.
