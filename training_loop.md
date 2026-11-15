## Explication de la training loop

On entraîne donc trois modèles en simultané :
- la politique
- le discriminateur (structure de GAIL)
- la critique V (car on fait du RL par PPO)


## Training Loop

RÉPÉTER jusqu'à avoir utilisé 100 à 300 millions d'échantillons :

  ── ÉTAPE 1 : LA POLITIQUE JOUE (aucun réseau n'apprend ici) ──────────
  Placer chaque personnage dans une pose prise au hasard dans la mocap.
  Tirer un objectif au hasard (une direction, une cible…).
  Tant que l'épisode n'est pas fini :
      La politique regarde l'état et l'objectif, et choisit des angles cibles.
      Le simulateur fait bouger le personnage avec les contrôleurs PD.
      On note l'état avant et l'état après le mouvement.
      La tâche donne une note : « est-ce que tu t'es rapproché du but ? »
      Le discriminateur donne une note : « est-ce que ça ressemble à la mocap ? »
      La récompense = moitié note de tâche + moitié note de style.
      Si le personnage tombe, l'épisode s'arrête.
  On garde tout en mémoire, et on range les transitions dans le replay buffer.

  ── ÉTAPE 2 : LE DISCRIMINATEUR APPREND ─────────────────────────────
  Prendre un paquet de transitions de la mocap         → elles doivent valoir +1.
  Prendre un paquet de transitions du replay buffer    → elles doivent valoir −1.
  Corriger D pour qu'il sépare mieux les deux,
  en le gardant « lisse » sur la mocap (gradient penalty).

  ── ÉTAPE 3 : LE CRITIQUE APPREND ───────────────────────────────────
  À partir des récompenses obtenues à l'étape 1,
  corriger V pour qu'il prédise mieux la somme des récompenses à venir
  depuis chaque état.

  ── ÉTAPE 4 : LA POLITIQUE APPREND ──────────────────────────────────
  Pour chaque action jouée, calculer son « avantage » :
      « ça s'est mieux passé que ce que V prévoyait ? »
  Rendre plus probables les actions qui ont fait mieux que prévu,
  et moins probables celles qui ont fait moins bien,
  sans trop s'écarter de la politique d'avant (écrêtage de PPO).

  Recommencer à l'étape 1 avec la nouvelle politique et le nouveau D.

À LA FIN : on garde uniquement la politique. D et V sont jetés.