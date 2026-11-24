# Runs RL : AMP Target Heading

Série N = `configs/runN/` = `outputs/rl/amp_<tâche>_N` (série 1 : sans suffixe). Chaque série entraîne
Walk (9 clips), Run (47 clips) et Locomotion (56 clips) pendant 300 M échantillons. Détail des
paramètres : `configs/README.md`. Courbes : `results.ipynb`. Runs restant à faire : `trains_to_run.md`.

## Série 1 : AMP de l'article

**Process.** Le PPO et l'AMP de `rl/` sur l'environnement Target Heading. La récompense est
r = 0,5 r_G (tâche : avancer dans la direction et à la vitesse demandées) + 0,5 r_S (style).
r_S = max(0, 1 − 0,25 (D − 1)²), avec D un discriminateur LSGAN qui compare les transitions
(Φ(s), Φ(s′)) de l'agent à celles de la mocap. Réglages de l'article et de MimicKit : pénalité de
gradient 10, 32 mises à jour de D par itération, entrée de D normalisée sur la mocap seule.

**Résultats et limites.** La tâche est apprise partout : retour normalisé de 0,82 (Walk) à 0,89 (Run).
Walk et Locomotion gardent un style informatif (r_S ≈ 0,22 et 0,27). Sur **Run, r_S s'effondre** : il
passe de 0,28 à 20 M échantillons à 0,09 à 50 M et 0,04 en fin de run. D reconnaît presque toute
transition de l'agent (D ≈ −1, précision de D sur l'agent ≈ 1), donc r_S n'a plus de dispersion et ne
guide plus le style. Première hypothèse : D est trop entraîné.

## Série 2 : D entraîné 4 fois moins

**Process.** `disc_steps` passe de 32 à 8. Chaque paire de l'agent du replay buffer est vue environ
8 fois par D au lieu de 30, pour que D ne surapprenne pas les transitions récentes de l'agent.

**Résultats et limites.** L'effondrement sur Run est seulement retardé : r_S vaut 0,20 à 50 M contre
0,09, mais 0,07 en fin de run, et D finit à −0,97. La tâche reste apprise (0,87). Walk a été arrêté à
79 M (r_S comparable à la série 1) et Locomotion n'a pas été lancé. D trouve donc toujours un critère
fiable : on essaie de le régulariser.

## Série 3 : D régularisé comme dans MimicKit

**Process.** Série 2 + la régularisation de D de l'AMP de MimicKit (absente de l'article) : un weight
decay de 1e-4 et une pénalité de 0,01 sur les poids de la dernière couche de D.

**Résultats et limites.** Aucun effet au-delà de la série 2. Sur Run, r_S vaut 0,18 à 50 M et 0,07 en
fin de run. Walk (r_S 0,27) et Locomotion (0,32) restent bons. L'explication est mécanique : avec SGD et
un pas de 1e-5, le weight decay retire 1e-9 par pas, la pénalité vaut environ 0,004, et la perte LSGAN
borne déjà les sorties de D. Hypothèse suivante : l'agent change trop vite pour que D le suive.

## Série 4 : pas de la politique de l'article, pieds alignés

**Process.** Série 2 + le pas de la politique de l'article (4e-6 au lieu des 5e-5 de MimicKit). On ajoute
aussi une correction de la mocap : chaque clip est décalé verticalement pour que ses pieds d'appui
soient au niveau de contact de la simulation. Les pieds de la mocap Run étaient 2 à 3 cm sous le sol,
ce qui faisait sauter le personnage au départ des épisodes.

**Résultats et limites.** Run a été arrêté à 85 M. La tâche est beaucoup plus lente (retour 0,01 à
85 M, contre un décollage vers 75 M auparavant), mais D sépare l'agent au même rythme (D ≈ −0,9) :
la vitesse de la politique n'est pas la cause. Un diagnostic par groupe de features désigne le cou et le
tronc. Dans les 47 clips Run, **le cou vaut exactement 0** (rotation et vitesse) sur toutes les frames,
déjà dans les fichiers de MimicKit. Comme l'entrée de D est normalisée sur la mocap seule, ces 18
entrées de Φ ont un écart-type au plancher (1e-4). Le moindre mouvement du cou de l'agent (1 à 2°) donne
alors ±10 en entrée de D : un indice parfait, que l'agent ne peut pas effacer. Walk et Locomotion ont
des clips au cou mobile, d'où l'absence du problème.

## Série 5 : entrées constantes masquées

**Process.** Série 2 + pieds alignés + `disc_mask_constant` : les entrées de Φ dont l'écart-type sur la
mocap est inférieur à 1e-3 sont mises à 0 pour D, côté mocap comme côté agent. Sur Run, cela retire les
18 entrées du cou. Sur Walk et Locomotion, une seule entrée est masquée, et elle est constante pour
l'agent aussi. D doit juger le style sur le reste du corps, qui est imitable.

**Résultats et limites.** En cours. Critère : r_S ne doit pas s'effondrer vers 0,05 entre 13 et 90 M
sur Run, et Walk et Locomotion doivent rester au niveau de la série 2. Une différence de style réelle
subsiste : l'abdomen de l'agent tourne plus que celui de la mocap (−25° à +40° contre ±11°). C'est le
prochain indice que D pourrait exploiter.

## Série 6 : normaliseur de D sur agent + mocap

**Process.** Série 2 + pieds alignés + `disc_norm_agent` : la moyenne et l'écart-type de l'entrée de D
sont mis à jour à chaque itération avec les paires de l'agent et autant de paires mocap, comme dans
MimicKit, puis figés après 1e8 échantillons. C'est l'autre correction du cou : son écart-type devient
celui des mouvements de l'agent (environ 1°), donc un cou à 1° donne une entrée d'environ 1 au lieu de
10. L'indice reste visible pour D, mais faible.

**Résultats et limites.** À venir. Avec la série 5, elle dira s'il faut retirer l'indice (masque) ou
s'il suffit de le ramener à une échelle normale (normalisation de MimicKit).
