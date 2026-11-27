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

**Validation.** Retour normalisé 0,80 (Walk), 0,88 (Run), 0,89 (Locomotion), et 0,93 pour le PPO seul
(`ppo_heading`). Une seule chute (1/32 épisodes, Run). Run et Locomotion atteignent les vitesses de 1 à
5 m/s à 0,5 m/s près. Walk échoue à 2 et 3 m/s (1,36 et 2,08 m/s atteints), comme en série 3 : les
clips Walk ne dépassent pas 1,5 m/s environ, et le style freine la politique au-delà.

## Série 2 : D entraîné 4 fois moins

**Process.** `disc_steps` passe de 32 à 8. Chaque paire de l'agent du replay buffer est vue environ
8 fois par D au lieu de 30, pour que D ne surapprenne pas les transitions récentes de l'agent.

**Résultats et limites.** L'effondrement sur Run est seulement retardé : r_S vaut 0,20 à 50 M contre
0,09, mais 0,07 en fin de run, et D finit à −0,97. La tâche reste apprise (0,87). D trouve donc toujours
un critère fiable : on essaie de le régulariser.

Walk et Locomotion ont été faits après les séries 5 et 6. **Walk plafonne** comme en séries 5 et 6 :
retour de test final 284 sur 600 (490 en séries 1 et 3), marche à ~1,3 m/s quelle que soit la cible.
Il n'a pas les pieds alignés : l'alignement n'est donc pas la cause du plafond. Les trois Walk qui
plafonnent (séries 2, 5, 6) ont `disc_steps` 8 sans régularisation de D, les deux qui n'en ont pas
(séries 1 et 3) ont `disc_steps` 32 ou la régularisation. Mais cette régularisation est numériquement
négligeable (série 3) et tous les runs ont la même graine : le hasard reste possible. Locomotion est
normal (retour de test final 513, r_S 0,34).

**Validation.** `amp_run_2` : retour normalisé 0,90, aucune chute, vitesses cibles de 1 à 5 m/s
atteintes à 0,4 m/s près. Ce suivi presque parfait vient de r_S éteint : la politique n'optimise que la
tâche. `amp_walk_2` : 0,50, aucune chute, 1,2 à 1,4 m/s pour toutes les cibles. `amp_locomotion_2` :
0,85, aucune chute, échoue à 1 et 2 m/s (0,44 et 1,43 m/s), atteint 3 à 5 m/s.

## Série 3 : D régularisé comme dans MimicKit

**Process.** Série 2 + la régularisation de D de l'AMP de MimicKit (absente de l'article) : un weight
decay de 1e-4 et une pénalité de 0,01 sur les poids de la dernière couche de D.

**Résultats et limites.** Aucun effet au-delà de la série 2. Sur Run, r_S vaut 0,18 à 50 M et 0,07 en
fin de run. Walk (r_S 0,27) et Locomotion (0,32) restent bons. L'explication est mécanique : avec SGD et
un pas de 1e-5, le weight decay retire 1e-9 par pas, la pénalité vaut environ 0,004, et la perte LSGAN
borne déjà les sorties de D. Hypothèse suivante : l'agent change trop vite pour que D le suive.

**Validation.** Retour normalisé 0,81 (Walk), 0,90 (Run), 0,88 (Locomotion), aucune chute. Walk
échoue au test de vitesse : 2,44 m/s pour 3 m/s demandés (seuil 0,5 m/s). Les clips Walk ne vont pas
au-delà de 1,5 m/s environ, et le style freine la politique entre marche et course. Le seuil a été
fixé pour le PPO seul : pour AMP, on lit le chiffre plutôt que le verdict.

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

Les trois tâches ont ensuite été relancées jusqu'à 300 M : **aucune n'apprend la tâche**. Les épisodes
de test passent de ~30 à ~70 pas sur 600 : le personnage commence à peine à tenir debout. Le pas de
l'article est trop lent pour notre budget de 300 M (la série 2 décolle vers 75–100 M avec 5e-5). Sur
Run, r_S s'effondre en plus (0,07, D −0,97), le cou n'étant pas corrigé.

**Validation.** Retour normalisé 0,02 (Run), 0,02 (Walk), 0,03 (Locomotion) : 32 chutes sur 32, en
1,5 à 2,2 s.

## Série 5 : entrées constantes masquées

**Process.** Série 2 + pieds alignés + `disc_mask_constant` : les entrées de Φ dont l'écart-type sur la
mocap est inférieur à 1e-3 sont mises à 0 pour D, côté mocap comme côté agent. Sur Run, cela retire les
18 entrées du cou. Sur Walk et Locomotion, une seule entrée est masquée, et elle est constante pour
l'agent aussi. D doit juger le style sur le reste du corps, qui est imitable.

**Résultats et limites.** Sur Run, **r_S ne s'effondre plus** : il reste entre 0,40 et 0,54 sur tout
le run (0,04 à 0,07 en séries 1 à 3), et D sur l'agent reste vers −0,55 au lieu de −1. D reconnaît
encore l'agent à 99 %, sans saturer : r_S garde sa dispersion et guide le style. Une différence réelle
subsiste, l'abdomen de l'agent tourne plus que celui de la mocap (−25° à +40° contre ±11°).

Sur **Walk, la tâche plafonne** : elle décolle à 118 M (85 à 92 M avant) et le retour de test stagne
vers 250–280 sur 600 (490 en séries 1 et 3). D et r_S sont normaux (r_S ≈ 0,35, D sur l'agent ≈ −0,7).
La politique reste dans un optimum local : elle marche à ~1,3 m/s, l'allure des clips Walk, quelle que
soit la vitesse demandée. `amp_walk_6` (sans masque) et `amp_walk_2` (sans masque ni pieds alignés)
plafonnent de la même façon : ni le masque ni l'alignement ne sont en cause (voir la série 2).

Locomotion est identique à la série 3 : décollage à 79 M, retour de test final 521 (523), r_S 0,36, D
sur l'agent −0,66. Les clips Run de Locomotion donnent sans doute à l'agent un style pour aller vite.

**Validation.** `amp_run_5` : retour normalisé 0,83 (0,90 en séries 2 et 3), aucune chute. Le test de
vitesse échoue à 2 m/s (1,50 m/s atteint), et toutes les vitesses sont un peu sous la cible. C'est le
compromis attendu d'AMP : le style est actif et retient la politique près de la mocap, au prix d'un
peu de tâche. Le creux à 2 m/s tomberait entre marche et course (hypothèse). À confirmer sur les vidéos.
`amp_walk_5` : retour normalisé 0,48, aucune chute, vitesse de 1,1 à 1,3 m/s pour toutes les cibles.
`amp_locomotion_5` : retour normalisé 0,87 (0,88 en série 3), aucune chute, toutes les vitesses
atteintes à 0,5 m/s près.

## Série 6 : normaliseur de D sur agent + mocap

**Process.** Série 2 + pieds alignés + `disc_norm_agent` : la moyenne et l'écart-type de l'entrée de D
sont mis à jour à chaque itération avec les paires de l'agent et autant de paires mocap, comme dans
MimicKit, puis figés après 1e8 échantillons. C'est l'autre correction du cou : son écart-type devient
celui des mouvements de l'agent (environ 1°), donc un cou à 1° donne une entrée d'environ 1 au lieu de
10. L'indice reste visible pour D, mais faible.

**Résultats et limites.** Sur Run, **r_S ne s'effondre pas non plus** : il reste entre 0,36 et 0,48
(0,41 en fin de run), et D sur l'agent reste vers −0,5, comme en série 5. Ramener l'indice du cou à une
échelle normale suffit donc, sans le retirer : la normalisation de MimicKit corrige seule le problème
de Run.

Sur **Walk, même plafond qu'en série 5** : la tâche décolle tôt (59 M), mais le retour de test reste
vers 250–300 sur 600 jusqu'à la fin (258). Le style domine encore plus : r_S 0,59 et D sur l'agent
−0,32, pour r_G 0,35. Le plafond n'est pas dû à la série 6 : `amp_walk_2` plafonne aussi (série 2).

Locomotion est au niveau des séries 3 et 5 pour la tâche (retour de test final 517, contre 521–523), avec
un style plus fort : r_S 0,48 (0,36) et D sur l'agent −0,46 (−0,66).

**Validation.** `amp_run_6` : retour normalisé 0,86, aucune chute, toutes les vitesses atteintes à
0,5 m/s près (0,85 à 4,72 m/s pour 1 à 5 m/s, un peu sous la cible). `amp_walk_6` : retour normalisé
0,45, aucune chute, 1,0 à 1,1 m/s pour toutes les cibles de 1 à 5 m/s. `amp_locomotion_6` : retour
normalisé 0,86, aucune chute, le test de vitesse échoue seulement à 2 m/s (1,46 m/s), le même creux
qu'`amp_run_5`.
