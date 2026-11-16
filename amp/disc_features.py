"""Features Φ(s) du discriminateur AMP (phase 1, article section 5.3).

Pour un état s, Φ(s) concatène, dans cet ordre :

    bloc                                    dim (humanoïde MimicKit)
    hauteur de la racine                     1
    rotation de la racine, 6D tan-norm       6     dans le repère du cap
    rotations locales des articulations     52     6D pour les 8 sphériques, angle pour les 4 révolutions
    positions des extrémités                12     mains + pieds, relatives à la racine, repère du cap
    vitesse linéaire de la racine            3     repère du cap
    vitesse angulaire de la racine           3     repère du cap
    vitesses articulaires                   28     (ablation : retirer ce bloc)
                                           ---
                                           105

Le « repère du cap » est la rotation de lacet seule (autour de z) de la racine.
Pour une transition (s, s'), on utilise le cap de s' pour les deux états
(cf. CADRAGE.md, phase 1) : voir `phi_paire`.

Conventions MimicKit : quaternions (x, y, z, w), z vertical, `joint_rot` de forme
(..., nb_joints - 1, 4) comme renvoyé par `KinCharModel.dof_to_rot` ou
`MotionLib.calc_motion_frame`. Le module doit être importé avec `mimickit/` dans
le sys.path (comme le reste du code MimicKit).
"""

from dataclasses import dataclass

import torch

import util.torch_util as torch_util
from anim.kin_char_model import JointType

EXTREMITES_DEFAUT = ("right_hand", "left_hand", "right_foot", "left_foot")


@dataclass
class EtatPerso:
    """Un état du personnage, en coordonnées monde. Dimensions de tête (...) libres."""
    root_pos: torch.Tensor       # (..., 3)
    root_rot: torch.Tensor       # (..., 4)  quaternion xyzw
    root_vel: torch.Tensor       # (..., 3)
    root_ang_vel: torch.Tensor   # (..., 3)
    joint_rot: torch.Tensor      # (..., J-1, 4)  rotation locale de chaque articulation
    dof_vel: torch.Tensor        # (..., nb_dof)
    key_pos: torch.Tensor        # (..., K, 3)  positions monde des extrémités


class AMPFeatures:
    """Calcule Φ(s) pour un personnage MimicKit donné."""

    def __init__(self, kin_char_model, extremites=EXTREMITES_DEFAUT, vitesses_articulaires=True):
        self.vitesses_articulaires = vitesses_articulaires
        self.key_body_ids = [kin_char_model.get_body_id(n) for n in extremites]

        # Index dans joint_rot (articulation j -> ligne j - 1), séparés par type.
        self.ids_spheriques, self.ids_revolutions, axes = [], [], []
        for j in range(1, kin_char_model.get_num_joints()):
            joint = kin_char_model.get_joint(j)
            if joint.joint_type == JointType.SPHERICAL:
                self.ids_spheriques.append(j - 1)
            elif joint.joint_type == JointType.HINGE:
                self.ids_revolutions.append(j - 1)
                axes.append(torch.as_tensor(joint.axis, dtype=torch.float32))
            # FIXED (mains) : aucun degré de liberté, rien à observer
        self.axes_revolutions = torch.stack(axes) if axes else torch.zeros(0, 3)
        self.nb_dof = kin_char_model.get_dof_size()

    @property
    def dim(self):
        d = 1 + 6 + 6 * len(self.ids_spheriques) + len(self.ids_revolutions)
        d += 3 * len(self.key_body_ids) + 3 + 3
        if self.vitesses_articulaires:
            d += self.nb_dof
        return d

    def extraire_key_pos(self, body_pos):
        """body_pos (..., nb_corps, 3) -> positions des extrémités (..., K, 3)."""
        return body_pos[..., self.key_body_ids, :]

    def __call__(self, etat: EtatPerso, cap_inv: torch.Tensor) -> torch.Tensor:
        """Φ(etat), exprimé dans le repère défini par `cap_inv` (quaternion (..., 4)
        qui ramène le monde dans le repère du cap). Renvoie (..., dim)."""
        hauteur = etat.root_pos[..., 2:3]

        rot_racine = torch_util.quat_mul(cap_inv, etat.root_rot)
        rot_racine = torch_util.quat_to_tan_norm(rot_racine)

        q_sph = etat.joint_rot[..., self.ids_spheriques, :]
        rot_sph = torch_util.quat_to_tan_norm(q_sph).flatten(-2)

        q_rev = etat.joint_rot[..., self.ids_revolutions, :]
        axes = self.axes_revolutions.to(q_rev).expand(q_rev.shape[:-1] + (3,))
        angles_rev = torch_util.quat_twist_angle(q_rev, axes)

        rel = etat.key_pos - etat.root_pos.unsqueeze(-2)
        cap_inv_k = cap_inv.unsqueeze(-2).expand(rel.shape[:-1] + (4,))
        extremites = torch_util.quat_rotate(cap_inv_k, rel).flatten(-2)

        v_lin = torch_util.quat_rotate(cap_inv, etat.root_vel)
        v_ang = torch_util.quat_rotate(cap_inv, etat.root_ang_vel)

        blocs = [hauteur, rot_racine, rot_sph, angles_rev, extremites, v_lin, v_ang]
        if self.vitesses_articulaires:
            blocs.append(etat.dof_vel)
        return torch.cat(blocs, dim=-1)

    def phi_paire(self, s: EtatPerso, s_suiv: EtatPerso) -> torch.Tensor:
        """Entrée du discriminateur (Φ(s), Φ(s')), les deux dans le repère du cap de s'.
        Renvoie (..., 2 * dim)."""
        cap_inv = torch_util.calc_heading_quat_inv(s_suiv.root_rot)
        return torch.cat([self(s, cap_inv), self(s_suiv, cap_inv)], dim=-1)
