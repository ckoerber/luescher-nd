"""Benchmarks eigenvalue computation
"""
import cProfile
import pstats
import argparse

import scipy.sparse as sp

from luescher_nd.solver import get_eigs

PARSER = argparse.ArgumentParser(description="Benchmark of eigenvalue computation")
PARSER.add_argument(
    "--n-order",
    "-n",
    dest="order",
    type=int,
    default=10000,
    help="Order of square matrix",
)
PARSER.add_argument(
    "--density",
    "-d",
    dest="density",
    type=float,
    default=0.05,
    help="Roughly number of entries in matrix in percent",
)
PARSER.add_argument(
    "--outfile",
    "-o",
    dest="outfile",
    type=str,
    default="outfile.prof",
    help="Outputfile for writing profile",
)
PARSER.add_argument(
    "--n-eigs",
    "-e",
    dest="n_eigs",
    type=int,
    default=20,
    help="Number of eigenvalues to extract. Is `min(n_order // 3, n_eigs)`.",
)


def main(args):
    """Creates matrix and computes eivenvalues.
    """
    if args.density < 0 or args.density > 1:
        raise ValueError("Density must be between zero and one.")

    for arg in vars(args):
        print(f"[+] {arg}: {getattr(args, arg)}")

    n_eigs = min(args.order // 3, args.n_eigs)

    print("[+] Creating matrix")
    m_sci = sp.random(args.order, args.order, args.density)
    m_sci += m_sci.T

    print("[+] Starting profile")
    cProfile.runctx(
        "get_eigs(m_sci, n_eigs)",
        {"get_eigs": get_eigs, "m_sci": m_sci, "n_eigs": n_eigs},
        {},
        filename=args.outfile,
    )

    p = pstats.Stats(args.outfile)
    p.strip_dirs().sort_stats("time").print_stats()


if __name__ == "__main__":
    ARGS = PARSER.parse_args()
    main(ARGS)
