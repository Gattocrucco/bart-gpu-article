"""Compare bartz with other BART packages in terms of RMSE on simulated data."""

import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from dataclasses import replace
from gc import collect
from pathlib import Path
from time import perf_counter
from typing import Any

import bartz
import numpy as np
import polars as pl
from bartz.jaxext import split
from equinox import Module
from jax import block_until_ready, random
from jax import numpy as jnp
from jaxtyping import Array, Key
from rpy2 import robjects

from bart_gpu_article.rbartpackages import BART3, bartMachine, dbarts
from bart_gpu_article.scripts.benchmark import (
    Data,
    format_time,
    make_data,
    make_int_seed,
)


class Config(Module):
    """Configuration of the script."""

    nvec: tuple[int, ...]
    fixed_ntree: int | None = 200
    fixed_p: int | None = 100
    n_over_ntree: int | None = None
    n_over_p: int | None = None
    only_data: bool = False
    seed: int = 2026_01_24_16_54
    max_n_times_ntree: int = 2**24  # determined empirically on my laptop
    max_n_times_p: int = 2**27  # determined empirically on my laptop
    n_test: int = 1000


class Timer:
    """Context manager to time a code block."""

    time: float

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, *_):
        self.time = perf_counter() - self.start


def make_split_data(
    key: Key[Array, ""], n_train: int, n_test: int, p: int
) -> tuple[Data, Data]:
    """Generate training and test data."""
    data = make_data(key, n_train + n_test, p)
    train = replace(
        data,
        raw_X=data.raw_X[:, :n_train],
        quantized_X=data.quantized_X[:, :n_train],
        y=data.y[:n_train],
    )
    test = replace(
        data,
        raw_X=data.raw_X[:, n_train:],
        quantized_X=data.quantized_X[:, n_train:],
        y=data.y[n_train:],
    )
    return train, test


def run_barts(cfg: Config) -> dict[str, dict[str, list]]:
    """Run all BART packages on simulated data."""
    # random seed
    key = random.key(cfg.seed)

    results = {}

    for n in cfg.nvec:
        # determine ntree and p for this n
        ntree = (
            max(1, n // cfg.n_over_ntree)
            if cfg.fixed_ntree is None
            else cfg.fixed_ntree
        )
        p = max(1, n // cfg.n_over_p) if cfg.fixed_p is None else cfg.fixed_p
        if n * ntree > cfg.max_n_times_ntree or n * p > cfg.max_n_times_p:
            break
        print(f"\nn = {n:_}, ntree = {ntree:_}, p = {p:_}")

        # split random seed
        keys = split(key, 6)
        key = keys.pop()

        print("generate data...")
        train, test = make_split_data(keys.pop(), n, cfg.n_test, p)
        block_until_ready((train, test))

        if cfg.only_data:
            for name in "bartz", "BART", "dbarts", "bartMachine":
                result = results.setdefault(name, {})
                result.setdefault("n", []).append(n)
                result.setdefault("prior_var", []).append(train.prior_var.item())
                result.setdefault("pop_var", []).append(train.pop_var.item())
                result.setdefault("eps_var", []).append(train.eps_var.item())
            del train, test
            collect()
            continue

        barts = {}

        print("run bartz...")
        kw_bartz = dict(
            x_test=test.raw_X,
            sigest=1,
            usequants=False,
            numcut=100,
            nskip=1000,
            ndpost=1000,
            ntree=ntree,
            seed=keys.pop(),
        )
        with Timer() as timer:
            barts["bartz"] = bartz.BART.gbart(train.raw_X, train.y, **kw_bartz)
            block_until_ready(barts["bartz"])
        print(format_time(timer.time))

        print("run BART...")
        kw_BART = kw_bartz.copy()
        kw_BART.update(
            x_test=test.raw_X.T,
            rm_const=False,
            mc_cores=1,
            seed=make_int_seed(keys.pop()),
        )
        with Timer() as timer:
            barts["BART"] = BART3.mc_gbart(train.raw_X.T, train.y, **kw_BART)
        print(format_time(timer.time))

        print("run dbarts...")
        kw_dbarts = kw_bartz.copy()
        kw_dbarts.update(
            x_test=test.raw_X.T,
            seed=make_int_seed(keys.pop()),
            keeptrainfits=False,
            keeptrees=False,
        )
        with Timer() as timer:
            barts["dbarts"] = dbarts.bart(train.raw_X.T, train.y, **kw_dbarts)
        print(format_time(timer.time))

        print("run bartMachine...")
        # I can't configure the splitting grid with bartMachine
        kw_bartMachine = kw_bartz.copy()
        kw_bartMachine.pop("x_test")
        kw_bartMachine.pop("usequants")
        kw_bartMachine.pop("numcut")
        kw_bartMachine.update(
            num_trees=kw_bartMachine.pop("ntree"),
            num_burn_in=kw_bartMachine.pop("nskip"),
            num_iterations_after_burn_in=kw_bartMachine.pop("ndpost"),
            run_in_sample=False,
            sig_sq_est=kw_bartMachine.pop("sigest"),
            mem_cache_for_speed=False,  # set to False to use less memory
            seed=make_int_seed(keys.pop()),
        )
        with Timer() as timer:
            barts["bartMachine"] = bartMachine.bartMachine(
                pl.DataFrame(np.array(train.raw_X.T)),
                pl.Series(np.array(train.y)),
                **kw_bartMachine,
            )
            barts["bartMachine"].yhat_test_mean = barts["bartMachine"].predict(
                pl.DataFrame(np.array(test.raw_X.T))
            )
        print(format_time(timer.time))

        print("test...")
        rmses = {}
        for name, bart in barts.items():
            rmse = jnp.sqrt(jnp.mean(jnp.square(bart.yhat_test_mean - test.y)))
            rmse = float(rmse)
            print(f"{name} rmse={rmse:#.3g}")
            rmses[name] = rmse

        # save results
        for name, rmse in rmses.items():
            result = results.setdefault(name, {})
            result.setdefault("n", []).append(n)
            result.setdefault("prior_var", []).append(train.prior_var.item())
            result.setdefault("pop_var", []).append(train.pop_var.item())
            result.setdefault("eps_var", []).append(train.eps_var.item())
            result.setdefault("rmse", []).append(rmse)

        # free memory
        del train, test, barts
        collect()
        robjects.r("gc()")

    return results


def save_results(results: dict[str, dict[str, list]], cfg: Config):
    """Save results to a JSON file."""
    results_dir = Path("./results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # build list of dictionaries
    output_list = []
    for name, result in results.items():
        entry = {
            "package": name,
            "results": result,
        }
        if cfg.fixed_ntree is None:
            entry["n/ntree"] = cfg.n_over_ntree
        else:
            entry["ntree"] = cfg.fixed_ntree
        if cfg.fixed_p is None:
            entry["n/p"] = cfg.n_over_p
        else:
            entry["p"] = cfg.fixed_p
        output_list.append(entry)

    # determine filename suffix
    suffix = ""
    if cfg.fixed_ntree is None:
        suffix += "-highntree"
    if cfg.fixed_p is None:
        suffix += "-highp"

    # save list as json file
    output_path = results_dir / f"rmse{suffix}.json"
    print(f"write {output_path}...")
    with open(output_path, "w") as f:
        json.dump(output_list, f, indent=4)


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--high-ntree",
        action="store_true",
        help="set ntree such that n/ntree=8 (ntree scales with n)",
    )
    parser.add_argument(
        "-p",
        "--high-p",
        action="store_true",
        help="set p such that n/p=10 (p scales with n)",
    )
    parser.add_argument(
        "-n",
        "--max-log2-n",
        type=int,
        default=4,
        help="upper end (included) of the n range as log2(n)",
    )
    return parser.parse_args()


def args_to_config(args: Namespace) -> Config:
    """Convert command line arguments to a Config object."""
    cfg_kwargs: dict[str, Any] = {}
    if args.high_ntree:
        cfg_kwargs["fixed_ntree"] = None
        cfg_kwargs["n_over_ntree"] = 8
    if args.high_p:
        cfg_kwargs["fixed_p"] = None
        cfg_kwargs["n_over_p"] = 10
    cfg_kwargs["nvec"] = tuple(2**p for p in range(2, args.max_log2_n + 1))
    return Config(**cfg_kwargs)


def main():
    """Entry point."""
    args = parse_args()
    cfg = args_to_config(args)
    results = run_barts(cfg)
    save_results(results, cfg)


if __name__ == "__main__":
    main()
