"""Command line entry point: ``python -m hct_survival.cli train``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from hct_survival.config import Config, CVConfig, Paths
from hct_survival.pipeline import run, save


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hct-survival",
        description="Ensemble event-free survival prediction after HCT.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="fit the ensemble and write artifacts")
    train.add_argument("--root", type=Path, default=None, help="project root")
    train.add_argument("--folds", type=int, default=10)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--models",
        nargs="+",
        default=["lgbm", "xgb", "xgb_deep", "hgb", "rf"],
        help="subset of base learners to fit",
    )
    train.add_argument("--no-stratify", action="store_true")
    train.add_argument("--no-rank-average", action="store_true")
    train.add_argument("--no-weight-search", action="store_true")
    train.add_argument(
        "--leaky-target",
        action="store_true",
        help="fit the Kaplan-Meier transform on all rows (the original behaviour)",
    )
    train.add_argument("--gpu", action="store_true")
    train.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(getattr(args, "verbose", False))

    if args.command == "train":
        paths = Paths(root=args.root) if args.root else Paths()
        config = Config(
            paths=paths,
            cv=CVConfig(
                n_splits=args.folds,
                random_state=args.seed,
                stratify=not args.no_stratify,
            ),
            fold_safe_target=not args.leaky_target,
            rank_average=not args.no_rank_average,
            optimise_weights=not args.no_weight_search,
            seeds=[args.seed],
            use_gpu=args.gpu,
            models=list(args.models),
        )
        result = run(config)
        save(result)

        print("\n=== Model comparison ===")
        print(result.leaderboard.to_string(index=False, float_format="%.4f"))
        print("\n=== Equity by race group (ensemble) ===")
        print(result.equity.to_string(index=False, float_format="%.4f"))
        print(f"\nEquity score: {result.summary['equity_score']:.4f}")
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
