# Réimplémentation d'AMP

Projet MVA : réimplémentation de *AMP: Adversarial Motion Priors for Stylized
Physics-Based Character Control* (Peng et al., SIGGRAPH 2021), dont le PDF est
`project-article.pdf`.

## Données

Les clips de mocap et le modèle du personnage viennent de
[MimicKit](https://github.com/xbpeng/MimicKit), le framework publié par Xue Bin
Peng, premier auteur d'AMP. On n'en garde que ce qui sert à la tâche Target
Heading :

| Jeu (Table 2 de l'article) | Fichiers | Clips | Durée |
|---|---|---|---|
| Walk | `humanoid_walk.pkl`, `long/humanoid_long_walk*.pkl` | 9 | 229,4 s |
| Run | `humanoid_run.pkl`, `sie/humanoid_sie_run_*.pkl` | 47 | 204,4 s |
| Locomotion | Walk + Run | 56 | 433,8 s |

```
data/
├── assets/humanoid/humanoid.xml   # personnage (MJCF, 34 DoF, axe z vertical)
├── motions/humanoid/              # clips au format MimicKit (.pkl)
│   ├── humanoid_walk.pkl
│   ├── humanoid_run.pkl
│   ├── long/                      # 4 longues marches et leurs miroirs
│   └── sie/                       # 46 courses (Sony Interactive Entertainment)
└── LICENSE_MimicKit
```

### Télécharger les données

`data/` n'est pas versionné (voir `.gitignore`). Pour obtenir les données,
lancer depuis la racine du projet (l'archive complète fait environ 400 Mo) :

```bash
TMP=$(mktemp -d)
URL="https://1sfu-my.sharepoint.com/:u:/g/personal/xbpeng_sfu_ca/EclKq9pwdOBAl-17SogfMW0Bved4sodZBQ_5eZCiz9O--w?e=bqXBaa"
# SharePoint refuse le téléchargement direct : une première requête récupère les cookies de session.
curl -sSL -c "$TMP/cookies" -b "$TMP/cookies" -o /dev/null "$URL"
curl -SL -c "$TMP/cookies" -b "$TMP/cookies" -o "$TMP/MimicKit_Data.zip" "$URL&download=1"
unzip -q "$TMP/MimicKit_Data.zip" \
  "MimicKit_Data/motions/humanoid/humanoid_walk.pkl" \
  "MimicKit_Data/motions/humanoid/humanoid_run.pkl" \
  "MimicKit_Data/motions/humanoid/long/humanoid_long_walk*" \
  "MimicKit_Data/motions/humanoid/sie/humanoid_sie_run_*" \
  "MimicKit_Data/motions/humanoid/sie/LICENSE.txt" \
  "MimicKit_Data/assets/humanoid/humanoid.xml" \
  "MimicKit_Data/LICENSE" -d "$TMP"
mkdir -p data && cp -R "$TMP/MimicKit_Data/motions" "$TMP/MimicKit_Data/assets" data/
cp "$TMP/MimicKit_Data/LICENSE" data/LICENSE_MimicKit
rm -rf "$TMP"
```

Si le lien ne marche plus, il est donné à l'étape 3 de l'installation dans le
[README de MimicKit](https://github.com/xbpeng/MimicKit#installation).

### À savoir sur ces clips

- **Format** : chaque `.pkl` est un dictionnaire `{"fps", "loop_mode", "frames"}`.
  Une frame compte 34 valeurs : position de la racine (3), rotation de la racine
  en exponential map (3), puis les 28 DoF des articulations dans l'ordre de
  `humanoid.xml`. Les clips sont à 120 Hz, sauf `humanoid_run.pkl` (30 Hz).
- **Ouverture** : un pickle peut exécuter du code quand on le charge. Ces fichiers
  ne contiennent que des types Python simples. On peut donc les lire avec un
  `pickle.Unpickler` dont `find_class` lève une exception.
- **Défaut connu** : `humanoid_walk.pkl` contient des NaN sur les 3 DoF du cou,
  sur toutes ses frames.
- **Miroirs** : les `*_mirror` sont des copies gauche/droite des longues marches,
  et une partie des courses `sie` semble l'être aussi.

### Licences

- Courses `sie/` : © 2021 Sony Interactive Entertainment, licence
  **CC BY-NC 4.0** (usage non commercial, citation obligatoire). Voir
  `data/motions/humanoid/sie/LICENSE.txt`.
- Reste des fichiers : licence Apache 2.0 de MimicKit (`data/LICENSE_MimicKit`).
