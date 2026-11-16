"""Validation de la phase 0 (voir Cadrage.md, section 4).

Trois tests, chacun en sous-commande :

  pretrained  rejoue `amp_steering_humanoid_model.pt` sous Newton et compare la durée
              des épisodes à celle de leur log (environ 300 pas).
  train       lance l'entraînement `amp_steering` de MimicKit pendant N minutes, puis
              appelle `compare` sur la sortie.
  compare     compare une sortie d'entraînement (stdout de run.py) à leur log
              `amp_steering_humanoid_log.txt` au même nombre d'échantillons, et
              mesure le débit (échantillons/s) et le temps projeté pour 100 M et 500 M.

Exemples (depuis la racine du projet, venv MimicKit activé) :

  python validation/phase0.py pretrained
  python validation/phase0.py train --minutes 20
  python validation/phase0.py compare outputs/phase0/train/train.out

MimicKit est cherché dans $MIMICKIT_DIR, sinon dans reference/MimicKit.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
MIMICKIT = Path(os.environ.get("MIMICKIT_DIR", PROJECT / "reference" / "MimicKit")).resolve()
REF_LOG = MIMICKIT / "data" / "logs" / "amp_steering_humanoid_log.txt"
MODEL = MIMICKIT / "data" / "models" / "amp_steering_humanoid_model.pt"
OUT = PROJECT / "outputs" / "phase0"

BASE_ARGS = [
    "--arg_file", "args/amp_steering_humanoid_args.txt",
    "--engine_config", "data/engines/newton_engine.yaml",
    "--visualize", "false",
]

TARGET_SAMPLES = 500e6   # Cadrage : environ 500 M d'échantillons...
TARGET_HOURS = 24.0      # ... en moins d'une journée


# --------------------------------------------------------------------------- logs

def read_table_log(path):
    """Lit un log texte de MimicKit (en-tête puis valeurs, séparés par des blancs)."""
    tokens = Path(path).read_text().split()
    ncol = next(i for i, t in enumerate(tokens) if _is_number(t))
    header, values = tokens[:ncol], tokens[ncol:]
    nrow = len(values) // ncol
    data = np.array(values[: nrow * ncol], dtype=float).reshape(nrow, ncol)
    return {k: data[:, i] for i, k in enumerate(header)}


def read_stdout_log(path):
    """Lit les tableaux `| clé | valeur |` affichés par run.py à chaque itération."""
    rows, row = [], {}
    for line in Path(path).read_text(errors="replace").splitlines():
        m = re.match(r"^\|\s*(\w+)\s*\|\s*(\S+)\s*\|$", line)
        if not m:
            continue
        key, val = m.groups()
        if key == "Iteration" and row:
            rows.append(row)
            row = {}
        if _is_number(val):
            row[key] = float(val)
    if row:
        rows.append(row)
    keys = set().union(*rows) if rows else set()
    return {k: np.array([r.get(k, np.nan) for r in rows]) for k in keys}


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _fmt_h(hours):
    return f"{hours:.1f} h" if hours >= 1 else f"{hours * 60:.0f} min"


# --------------------------------------------------------------------------- tests

def run_mimickit(extra_args, log_file, timeout_s=None):
    """Lance mimickit/run.py depuis le dossier de MimicKit ; stdout va dans log_file."""
    if not (MIMICKIT / "mimickit" / "run.py").exists():
        sys.exit(f"MimicKit introuvable dans {MIMICKIT} (définir MIMICKIT_DIR)")
    env = dict(os.environ)
    env.setdefault("WARP_CACHE_PATH", str(MIMICKIT.parent / "warpcache"))
    cmd = [sys.executable, "mimickit/run.py", *BASE_ARGS, *extra_args]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print("$", " ".join(cmd), f"> {log_file}")
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, cwd=MIMICKIT, env=env, stdout=f, stderr=subprocess.STDOUT)
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            print(f"arrêt après {timeout_s / 60:.0f} min (prévu)")
    return proc.returncode


def test_pretrained(args):
    out = OUT / "pretrained"
    run_mimickit(
        ["--mode", "test", "--num_envs", str(args.num_envs),
         "--test_episodes", str(args.episodes), "--model_file", str(MODEL),
         "--out_dir", str(out)],
        out / "test.out",
    )
    text = (out / "test.out").read_text(errors="replace")
    ret = re.search(r"Mean Return: (\S+)", text)
    ep_len = re.search(r"Mean Episode Length: (\S+)", text)
    if not (ret and ep_len):
        print(text[-2000:])
        sys.exit("échec : pas de résultat de test dans la sortie")
    ret, ep_len = float(ret.group(1)), float(ep_len.group(1))

    ref = read_table_log(REF_LOG)
    ref_len, ref_ret = ref["Test_Episode_Length"][-1], ref["Test_Return"][-1]
    ok = ep_len >= 0.8 * ref_len
    print("\n=== Modèle pré-entraîné sous Newton ===")
    print(f"  durée moyenne des épisodes : {ep_len:7.1f} pas   (leur log : {ref_len:.1f})")
    print(f"  retour moyen               : {ret:7.1f}       (leur log : {ref_ret:.1f})")
    print(f"  => {'OK' if ok else 'ÉCHEC'} (critère : au moins 80 % de leur durée)")
    return ok


def test_train(args):
    out = OUT / "train"
    run_mimickit(
        ["--mode", "train", "--num_envs", str(args.num_envs), "--out_dir", str(out)],
        out / "train.out",
        timeout_s=args.minutes * 60,
    )
    return compare(out / "train.out")


def compare(path):
    ours = read_stdout_log(path)
    if "Samples" not in ours or len(ours["Samples"]) < 2:
        sys.exit(f"pas assez d'itérations dans {path} (le démarrage prend plusieurs minutes : lancer au moins 10 min)")
    ref = read_table_log(REF_LOG)

    # Débit : on ignore la première itération (compilation des kernels).
    s, t = ours["Samples"], ours["Wall_Time"]
    sps = (s[-1] - s[0]) / ((t[-1] - t[0]) * 3600)
    ref_sps = (ref["Samples"][-1] - ref["Samples"][0]) / ((ref["Wall_Time"][-1] - ref["Wall_Time"][0]) * 3600)
    hours_target = TARGET_SAMPLES / sps / 3600

    print("\n=== Débit ===")
    print(f"  nous      : {sps:8.0f} échantillons/s  ({len(s)} itérations, {s[-1] / 1e6:.1f} M échantillons)")
    print(f"  leur log  : {ref_sps:8.0f} échantillons/s  (Isaac Gym)")
    print(f"  projection: 100 M en {_fmt_h(100e6 / sps / 3600)}, 500 M en {_fmt_h(hours_target)}")
    ok_speed = hours_target < TARGET_HOURS
    print(f"  => {'OK' if ok_speed else 'ÉCHEC'} (critère : 500 M en moins de {TARGET_HOURS:.0f} h)")

    print("\n=== Courbe d'apprentissage, comparée à leur log au même nombre d'échantillons ===")
    # Le stdout de run.py ne contient pas les durées d'épisodes, et les tests n'ont lieu
    # que toutes les `iters_per_output` itérations (200) : Train_Return est la seule
    # courbe disponible à chaque itération. Le log.txt du dossier de sortie a le reste.
    # (Test_Return y est répété tel quel entre deux tests : on ne l'affiche pas ici.)
    keys = ["Train_Return"]
    print(f"  {'échantillons':>12} " + " ".join(f"{k:>22}" for k in keys))
    n = len(s)
    for i in sorted(set(np.linspace(0, n - 1, min(n, 8)).astype(int))):
        cells = []
        for k in keys:
            ref_v = np.interp(s[i], ref["Samples"], ref[k])
            cells.append(f"{ours[k][i]:9.1f} (réf {ref_v:6.1f})")
        print(f"  {s[i] / 1e6:10.1f} M " + " ".join(f"{c:>22}" for c in cells))

    log_txt = Path(path).parent / "log.txt"
    if log_txt.exists():
        tab = read_table_log(log_txt)
        print(f"\n  Points de test ({log_txt.name}, toutes les 200 itérations) :")
        print(f"  {'échantillons':>12} {'Test_Episode_Length':>26} {'Test_Return':>22}")
        for j in range(len(tab["Samples"])):
            sj = tab["Samples"][j]
            print(f"  {sj / 1e6:10.1f} M "
                  f"{tab['Test_Episode_Length'][j]:13.1f} (réf {np.interp(sj, ref['Samples'], ref['Test_Episode_Length']):6.1f}) "
                  f"{tab['Test_Return'][j]:9.1f} (réf {np.interp(sj, ref['Samples'], ref['Test_Return']):6.1f})")

    takeoff = ref["Samples"][np.argmax(ref["Test_Episode_Length"] > 150)]
    if s[-1] < takeoff:
        print(f"\n  Comparaison non concluante : leur courbe décolle vers {takeoff / 1e6:.0f} M "
              f"échantillons, ce run s'arrête à {s[-1] / 1e6:.1f} M "
              f"(il faudrait environ {_fmt_h(takeoff / sps / 3600)}).")
    return ok_speed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("pretrained", help="rejouer le modèle pré-entraîné")
    a.add_argument("--num_envs", type=int, default=256)
    a.add_argument("--episodes", type=int, default=512)

    a = sub.add_parser("train", help="entraînement court + comparaison au log")
    a.add_argument("--minutes", type=float, default=20)
    a.add_argument("--num_envs", type=int, default=4096)

    a = sub.add_parser("compare", help="comparer une sortie d'entraînement existante")
    a.add_argument("stdout_file", type=Path)

    args = p.parse_args()
    t0 = time.time()
    if args.cmd == "pretrained":
        ok = test_pretrained(args)
    elif args.cmd == "train":
        ok = test_train(args)
    else:
        ok = compare(args.stdout_file)
    print(f"\n({time.time() - t0:.0f} s)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
