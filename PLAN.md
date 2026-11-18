# Plan : état d'avancement

Ce fichier suit la structure de `pseudo-code.py` et ne décrit que ce qui est fait (phases 0 et 1).
Pour les détails, voir `CADRAGE.md` et `validation/phase*.md`.

## 0. Configuration (phase 0) : fait

But : vérifier que MimicKit fonctionne sous le simulateur Newton avant d'écrire notre code.
Script : `validation/phase0.py`.

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

L'agent est lu dans Newton, alors que l'expert vient de la mocap (MimicKit). Les positions
coïncident, mais **pas les vitesses** :
- pour les articulations sphériques, MimicKit donne la vitesse angulaire dans le repère de
  l'enfant, Newton l'attend dans celui du parent (ω_parent = q · ω_enfant) ;
- pour la racine, MimicKit donne la vitesse de l'origine du bassin, Newton celle de son centre
  de masse (v_com = v_origine + ω × r).

Sans conversion, l'écart atteint 12 rad/s et 4 m/s. Le discriminateur pourrait alors distinguer
l'expert de l'agent par la seule convention des vitesses. La couche Newton de MimicKit ne fait
pas cette conversion, ce qui explique peut-être l'échec du modèle pré-entraîné (non vérifié).

- Fait : la conversion des transitions expertes (`mocap_velocities_to_newton` dans
  `amp/expert.py`).
- **À faire (phase 2)** : la même conversion au démarrage des épisodes depuis la mocap (RSI).

Autres pièges :
- après avoir écrit un état dans Newton, il faut appeler `eval_fk`, sinon Newton garde
  l'ancienne pose ;
- juste après `eval_fk`, `get_body_vel` renvoie des vitesses fausses et ne doit pas être lu ;
- le cou de `humanoid_walk.pkl` contient des NaN, que MimicKit remplace par une rotation nulle
  (le cou reste figé dans ce clip).

### Validation : `validation/phase1.py` (tous les tests OK)

- Test 1 : vérifie que Φ calculé sur la mocap est égal à Φ recalculé après avoir placé Newton
  dans le même état.
- Test 1b : vérifie que les vitesses de chaque corps sont les mêmes dans Newton et dans la mocap,
  après conversion.
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

`validation/phase1_view_clip.py` fait une vidéo d'un clip joué dans Newton pour contrôler
visuellement les poses et les vitesses. `notebooks/01_visualisation_mocap.ipynb` sert à explorer
les données.

## 2. Environnement Target Heading (phase 2) : à faire

Il faudra y appliquer la conversion des vitesses au démarrage des épisodes.

## 3. Réseaux : à faire

## 4. Boucle d'entraînement (phases 3 et 4) : à faire

## 5. Évaluation (phase 5) : à faire
