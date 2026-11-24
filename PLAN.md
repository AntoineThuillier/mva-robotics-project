# Plan : état d'avancement

Ce fichier ne décrit que ce qui est fait (phases 0 à 2).
Pour les détails, voir `CADRAGE.md` et `validation/README.md` (bilans `validation/check_*.md`).

## 0. Configuration (phase 0) : fait

But : vérifier que MimicKit fonctionne sous le simulateur Newton avant d'écrire notre code.
Script : `validation/check_mimickit_install.py`.

- `pretrained` : rejoue le modèle AMP pré-entraîné de MimicKit. **Échec** : les épisodes durent
  80 pas au lieu de 300. Le modèle a été entraîné sous Isaac Gym et ne se transfère pas à Newton.
- `train` : lance l'entraînement AMP de MimicKit. OK : la courbe décolle comme la leur.
- `compare` : compare notre log au leur et mesure le débit. OK : environ 8 000 échantillons/s,
  soit 500 M échantillons en 17 h environ.

Configuration retenue : Newton, physique à 240 Hz, politique à 30 Hz, 4096 environnements sur GPU.

## 1. Features du discriminateur Φ (phase 1) : fait

But : produire les exemples « vrais » du discriminateur, c'est-à-dire des transitions (s, s')
tirées de la mocap.

- `amp/disc_features.py` : la classe `AMPFeatures` calcule Φ(s), un vecteur de 105 nombres
  (hauteur et rotation de la racine, rotations des articulations, positions des mains et des
  pieds, vitesses). Tout est exprimé dans le repère du cap (rotation autour de z seule), donc Φ
  ne dépend ni de la position du personnage ni de la direction vers laquelle il regarde.
  `phi_paire(s, s')` renvoie `[Φ(s), Φ(s')]`, les deux états étant exprimés dans le repère du
  cap de s'.
- `amp/expert.py` : `ExpertTransitions.sample(n)` tire un clip selon les poids du dataset et un
  instant t, prend s à t et s' à t + 1/30 s, convertit les vitesses pour Newton, puis calcule
  `phi_paire`.

### Traduction MimicKit → Newton (nécessaire)

L'agent est lu dans Newton, alors que l'expert vient de la mocap (MimicKit). Les deux doivent
être dans la même convention, sinon le discriminateur pourrait distinguer l'expert de l'agent
par la seule convention des vitesses. **Une seule vitesse diffère** (mesuré dans le simulateur
en phase 2, `validation/check_heading_env.md`, test 2) :
- pour la racine, MimicKit donne la vitesse de l'origine du bassin, Newton celle de son centre
  de masse (v_com = v_origine + ω × r) ;
- pour les articulations sphériques, les deux utilisent le repère de l'enfant : **rien à
  convertir**. La phase 1 avait conclu qu'il fallait les tourner vers le repère du parent, parce
  qu'elle passait par `eval_fk`, qui suppose ce repère, alors que le solveur de Newton, qui fait
  la simulation, utilise celui de l'enfant.

La couche Newton de MimicKit ne convertit pas la vitesse de la racine.

- Fait : la conversion des transitions expertes, racine seulement (`mocap_velocities_to_newton`
  dans `amp/expert.py`, corrigé le 19 nov. 2026 : la rotation des articulations est retirée).
- Fait : la conversion au démarrage des épisodes (RSI), racine seulement (`amp/heading_env.py`).

Autres pièges :
- après avoir écrit un état dans Newton, il faut appeler `eval_fk`, sinon Newton garde
  l'ancienne pose ;
- juste après `eval_fk`, `get_body_vel` renvoie des vitesses fausses et ne doit pas être lu ;
- le cou de `humanoid_walk.pkl` contient des NaN, que MimicKit remplace par une rotation nulle
  (le cou reste figé dans ce clip).

### Validation : `validation/check_disc_features.py` (tous les tests OK)

- Test 1 : vérifie que Φ calculé sur la mocap est égal à Φ recalculé après avoir placé Newton
  dans le même état.
- Test 1b : documente un piège. `eval_fk` lit les vitesses des articulations sphériques dans le
  repère du parent, alors que le solveur les lit dans celui de l'enfant ; il ne faut donc pas
  utiliser les vitesses des corps calculées par `eval_fk`.
- Test 2 : vérifie le contenu de Φ bloc par bloc, en comparant les tailles à DeepMimic et les
  valeurs à MimicKit.
- Test 3 : trace la vitesse du bassin pour Walk (médiane de 0,9 m/s) et pour Run (médiane de
  2,4 m/s).
- Test 4 : vérifie que le nombre et la durée des clips correspondent à la Table 2 de l'article.
- Test 5 : vérifie que Φ ne change pas si on tourne le monde autour de z ou si on le déplace
  horizontalement.
- Test 6 : vérifie qu'il n'y a aucun NaN, ni dans la mocap ni dans Φ.
- Test 7 : vérifie le sampler `ExpertTransitions` de bout en bout : forme du résultat, paires
  dans un même clip, respect des poids, cohérence avec Newton.

`validation/video_mocap_clip.py` fait une vidéo d'un clip joué dans Newton pour contrôler
visuellement les poses et les vitesses. `notebooks/01_visualisation_mocap.ipynb` sert à explorer
les données.

## 2. Environnement Target Heading (phase 2) : fait

But : le monde dans lequel la politique agit. Le personnage doit aller dans une direction cible
à une vitesse cible ; l'environnement donne le goal, calcule la récompense de tâche r_G et
décide de la fin des épisodes.

- `amp/heading_env.py` : `TaskHeadingEnv` hérite de l'environnement steering de MimicKit
  (physique, PD, chutes, démarrage depuis la mocap) et ne change que :
  - le goal g = (direction cible vue depuis le personnage, v*), sans la direction de regard de
    MimicKit ;
  - son calendrier, celui de DeepMimic : toutes les 0,2 à 0,5 s, dérive du cap de N(0 ; 0,15),
    virage brusque avec une probabilité de 0,01, nouvelle v* dans [1, 5] m/s avec une
    probabilité de 0,02 ;
  - r_G (éq. 11) sur la vitesse moyenne du centre de masse pendant le pas, 0 si le personnage
    recule, sans le terme de regard de MimicKit ;
  - des épisodes de 20 s ;
  - la vitesse de la racine convertie au démarrage des épisodes.
- `configs/run1/locomotion_env.yaml` : sa configuration.
- `validation/load_envs.py` : construit les environnements des scripts de validation, une
  seule fois par processus.

### Validation : `validation/check_heading_env.py` (tous les tests OK)

- Test 1 : vérifie qu'avec des actions aléatoires le personnage tombe en moins de 3 s.
- Test 2 : vérifie les vitesses écrites au démarrage, mesure les conventions du simulateur
  (en l'air, sans PD, un pas de 1/240 s), et vérifie que le suivi de la mocap reste proche du
  clip après un pas. Sans politique, le personnage tombe en moins d'une seconde : le critère
  « quelques secondes » du cadrage a été remplacé.
- Test 3 : vérifie r_G contre l'éq. 11 calculée à la main, et le centre de masse utilisé.
- Test 4 : vérifie les statistiques du calendrier du goal contre celles de DeepMimic.
- Test 5 : vérifie l'observation g.
- Test 6 : vérifie que les épisodes s'arrêtent à 20 s.

## 3. Réseaux : à faire

## 4. Boucle d'entraînement (phases 3 et 4) : à faire

## 5. Évaluation (phase 5) : à faire
