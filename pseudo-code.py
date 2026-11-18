# =============================================================================
#  AMP (Peng et al., SIGGRAPH 2021) : réimplémentation de Target Heading dans MimicKit
#  Pseudo-code, non exécutable. Voir Cadrage.md pour les phases et les validations.
# -----------------------------------------------------------------------------
#  Légende :
#    [MIMICKIT]  réutilisé tel quel (moteur Newton, personnage, PD, motion_lib, PPO)
#    [NOUS]      écrit par nous : l'apport de l'article (Phi, D, r_S, tâche)
#    [ÉCART]     choix qui diverge d'une des sources (article / DeepMimic / MimicKit),
#                justifié dans Cadrage.md, section 5
# =============================================================================


# =============================================================================
# 0. CONFIGURATION  (section 8.1, Table 4)
# =============================================================================

MOTEUR       = "newton"     # [MIMICKIT][ÉCART] article : Bullet 1,2 kHz, CPU
SIM_FREQ     = 240          # Hz, défaut Newton dans MimicKit
CONTROL_FREQ = 30           # Hz, comme l'article -> 8 pas de physique par action
NUM_ENVS     = 4096         # envs vectorisés sur GPU (article : 16 workers CPU)
DUREE_EP     = 20.0         # s  [ÉCART] MimicKit : 10 s, DeepMimic : 20 s

W_TACHE, W_STYLE = 0.5, 0.5 # éq. 4  (extension : W_STYLE = 1.0, cf. RewardScale DeepMimic)
W_GP     = 10               # éq. 8
N_HIST_D = 2                # états vus par D [ÉCART] MimicKit : 10 (extension)
BUDGET   = 300e6            # échantillons par politique (article : 100 à 300 M)
GRAINES  = [0, 1, 2]

DATASETS = {                # [MIMICKIT] .pkl dans data/motions/humanoid/ (Table 2)
    "Walk":       ["humanoid_walk", "long/humanoid_long_walk*"],          #  9 clips, 229 s
    "Run":        ["humanoid_run", "sie/humanoid_sie_run_*"],             # 47 clips, 204 s
    "Locomotion": "Walk + Run",                                           # 56 clips, 434 s
}


# =============================================================================
# 1. FEATURES DU DISCRIMINATEUR Phi  (section 5.3)  -- [NOUS], phase 1
# =============================================================================

def compute_amp_features(etat, repere):
    # Tout est exprimé dans `repere` = repère du cap (yaw seul) de l'état t+1,
    # appliqué aux deux états de la paire.
    return concat(
        hauteur_racine(etat),
        rot6d(rotation_racine(etat), repere),       # normale-tangente
        rot6d(rotations_spheriques(etat)),          # angle seul pour les révolutions
        positions_extremites(etat, repere),         # mains + pieds, relatives à la racine
        vitesse_lin_racine(etat, repere), vitesse_ang_racine(etat, repere),
        vitesses_articulaires(etat),                # ablation : retirer les vitesses
    )

def phi_paire(s, s_suiv):
    repere = repere_cap(s_suiv)
    return concat(compute_amp_features(s, repere), compute_amp_features(s_suiv, repere))

def mocap_vers_newton(etat):                                    # [NOUS] -> amp/expert.py, amp/heading_env.py
    # Les états de l'agent sont lus dans Newton : toute vitesse qui vient de la mocap doit
    # être dans la convention de Newton, sinon D distingue expert et agent par la seule convention.
    # Une seule différence (mesurée dans le simulateur, validation/phase2.md, test 2) :
    etat.root_vel += cross(etat.root_ang_vel, rotation(etat.root_rot) @ COM_RACINE)  # origine -> CdM
    # Positions, rotations, vitesse angulaire de la racine : identiques.
    # Articulations sphériques : identiques aussi (repère de l'enfant des deux côtés). Ne PAS les
    # tourner : eval_fk de Newton suppose le repère du parent, mais le solveur, lui, utilise
    # celui de l'enfant. [À CORRIGER avant la phase 4] amp/expert.py les tourne encore.
    return etat                                                 # COM_RACINE ≈ (0, 0, 0,113), lu dans Newton

def sample_expert_transitions(M, n):                            # -> amp/expert.py, ExpertTransitions
    # [MIMICKIT] motion_lib : interpolation et vitesses ; [NOUS] le tirage et la conversion
    clip, t = M.tirer_clip_et_instant_continu(n, marge=1 / CONTROL_FREQ)   # s et s' dans le même clip
    s      = mocap_vers_newton(M.etat(clip, t))
    s_suiv = mocap_vers_newton(M.etat(clip, t + 1 / CONTROL_FREQ))
    return phi_paire(s, s_suiv)


# =============================================================================
# 2. ENVIRONNEMENT TARGET HEADING  (section 7, éq. 11)  -- phase 2
# =============================================================================
# [MIMICKIT] physique, PD (actions = cibles en carte exponentielle 3D / angle 1D),
#            RSI depuis la mocap, détection de chute.
# FAIT : amp/heading_env.py (TaskHeadingEnv hérite de task_steering_env.py de MimicKit),
#        config configs/amp_heading_humanoid_env.yaml, validation/phase2.py (6 tests OK).

def reset_rsi(envs, M):                                         # [MIMICKIT][NOUS] correction
    # RSI : l'épisode démarre dans un état tiré de la mocap. La couche Newton de MimicKit
    # y écrit la vitesse de la racine SANS conversion : on la corrige (_ref_state_init).
    clip, t = M.tirer_clip_et_instant_continu(len(envs))
    envs.ecrire_etat(mocap_vers_newton(M.etat(clip, t)))        # seule la racine change

def tirer_goal_initial(envs):                                   # [NOUS]
    envs.v_cible = uniforme(1, 5)                               # [ÉCART] MimicKit : [0,5 ; 5]
    envs.dir_cible = uniforme(-pi, pi)
    envs.prochaine_maj = envs.temps + uniforme(0.2, 0.5)

def mettre_a_jour_goal(envs):                                   # [NOUS][ÉCART] calendrier DeepMimic
    # quand envs.temps >= envs.prochaine_maj ; probabilités tirées à chaque mise à jour, pas à chaque pas
    if proba(0.01): envs.dir_cible += uniforme(-pi, pi)         # virage brusque
    else:           envs.dir_cible += normal(0, 0.15)           # dérive lente
    if proba(0.02): envs.v_cible = uniforme(1, 5)               # nouvelle vitesse
    envs.prochaine_maj = envs.temps + uniforme(0.2, 0.5)

def observer_goal(envs):
    delta = envs.dir_cible - cap(envs.etat)
    return (cos(delta), sin(delta), envs.v_cible)               # g, entrée de pi et V
    # DeepMimic écrit -sin(delta) dans son repère y vertical : même grandeur (direction cible
    # vue depuis le personnage), MimicKit étant en z vertical

def recompense_tache(envs, com_avant, com_apres):               # [NOUS][ÉCART] version DeepMimic
    v_moy = (com_apres - com_avant) * CONTROL_FREQ              # vitesse moyenne du CdM sur le pas
    v_proj = dot(d_cible(envs), v_moy)
    r_G = exp(-0.25 * (envs.v_cible - v_proj) ** 2)             # éq. 11
    return where(v_proj < 0, 0, r_G)                            # pas de terme de regard (MimicKit)

def fin_episode(envs):                                          # [MIMICKIT]
    return contact_sol_hors_pieds(envs) | (envs.temps > DUREE_EP)


# =============================================================================
# 3. RÉSEAUX  (section 6.2)
# =============================================================================
#   pi(a | s, g) : MLP 1024-512 ReLU, gaussienne de Sigma fixe   [MIMICKIT][ÉCART] pas de "gating"
#   V(s, g)      : MLP 1024-512                                  [MIMICKIT]
#   D(phi)       : MLP 1024-512, sortie linéaire (LSGAN), entrée normalisée   [NOUS]


# =============================================================================
# 4. BOUCLE D'ENTRAÎNEMENT  (Algorithme 1)  -- phase 3 (W_STYLE = 0) puis phase 4
# =============================================================================

for nom in ["Walk", "Run", "Locomotion"]:
    for graine in GRAINES:
        M = charger_mocap(DATASETS[nom])
        envs = creer_envs_heading(NUM_ENVS, M)
        pi, V, D = MLP(), MLP(), MLP()
        B = ReplayBuffer(taille=1e5)                            # transitions de la politique

        n = 0
        while n < BUDGET:
            # (a) COLLECTE : NUM_ENVS x K pas, reset auto par RSI   [MIMICKIT] + conversion, voir reset_rsi
            for t in range(K):
                a = pi.echantillonner(s, g)
                s_suiv = envs.pas(a)                            # 8 pas Newton
                phi = phi_paire(s, s_suiv)                      # s, s_suiv lus dans Newton : déjà dans sa convention

                r_G = recompense_tache(envs, ...)
                r_S = max(0, 1 - 0.25 * (D(phi) - 1) ** 2)     # éq. 7     [NOUS]
                r   = W_TACHE * r_G + W_STYLE * r_S             # éq. 4     [NOUS]

                stocker(s, g, a, r, fin_episode(envs))
                B.ajouter(phi)
                n += NUM_ENVS

            # (b) DISCRIMINATEUR  (éq. 8)   [NOUS]  lr = 1e-5
            for _ in range(nb_pas_D):
                vrai = sample_expert_transitions(M, 256)        # cible +1
                faux = B.tirer(256)                             # cible -1
                perte_D = ( moyenne((D(vrai) - 1) ** 2)
                          + moyenne((D(faux) + 1) ** 2)
                          + W_GP / 2 * moyenne(norme(grad(D, vrai)) ** 2) )  # GP sur la mocap seule
                D.descente_gradient(perte_D)
                # si D instable : weight decay 5e-4 + pénalité sur la dernière couche (DeepMimic)

            # (c) PPO   [MIMICKIT]
            #   gamma = 0,99 ; GAE(0,95) ; TD(0,95) ; minibatch 256
            #   clip = 0,2  [ÉCART] Table 4 indique 0,02, probablement une coquille
            #   taille de batch et lr à réajuster pour 4096 envs GPU (Table 4 : 4096 éch./itération)
            ppo_update(pi, V, donnees)

        # (d) DIAGNOSTICS pendant l'entraînement
        #   précision de D sur vrai / faux, r_S moyen, valeur du GP, durée des épisodes


# =============================================================================
# 5. ÉVALUATION  -- phase 5
# =============================================================================
#   Retour de tâche normalisé, 3 graines x 32 épisodes -> Table 1
#       article : Locomotion 0,90 ; Walk 0,46 ; Run 0,63
#   Vitesse atteinte en fonction de v*                -> Fig. 4 (droite)
#   Courbes d'apprentissage des 3 jeux                -> Fig. 4 (gauche)
#   Comparaisons : notre AMP, l'AMP de MimicKit, PPO seul ("No Data", Fig. 5)
#
#   Extensions : W_STYLE = 1,0 ; N_HIST_D = 10 ; ablations sans GP / sans vitesses
#                dans Phi ; métrique de style (glissement des pieds, distance à la mocap)
# =============================================================================
