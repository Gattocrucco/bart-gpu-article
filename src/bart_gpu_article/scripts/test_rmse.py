"""Compare bartz with other BART packages in terms of RMSE on simulated data."""

import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from gc import collect
from math import ceil
from pathlib import Path
from subprocess import PIPE, TimeoutExpired, run
from sys import argv, executable, stderr
from time import perf_counter
from types import MappingProxyType
from typing import Any

import bartz
import numpy as np
import polars as pl
from bartz.jaxext import split
from equinox import Module
from jax import block_until_ready, random
from jaxtyping import Array, Float32, Float64, Key
from rpy2 import robjects
from wurlitzer import pipes

from bart_gpu_article.rbartpackages import BART3, bartMachine, dbarts
from bart_gpu_article.scripts.benchmark import (
    Data,
    format_time,
    make_data,
    make_int_seed,
)


class Config(Module):
    """Configuration of the script."""

    seed: int
    nvec: tuple[int, ...]
    method: str | None
    fixed_ntree: int | None = 200
    fixed_p: int | None = 100
    n_over_ntree: int | None = None
    n_over_p: int | None = None
    n_test: int = 1000
    timeout: float = 120.0  # seconds


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
    data = make_data(key, n_train + n_test, p, quantized_x=False)
    train = replace(
        data,
        raw_X=data.raw_X[:, :n_train],
        quantized_X=None,
        y=data.y[:n_train],
    )
    test = replace(
        data,
        raw_X=data.raw_X[:, n_train:],
        quantized_X=None,
        y=data.y[n_train:],
    )
    return train, test


def get_bartz_kwargs(ntree: int, test: Data) -> Mapping[str, Any]:
    """Get the keyword arguments for bartz."""
    return MappingProxyType(
        dict(
            x_test=test.raw_X,
            sigest=1,
            usequants=False,
            numcut=100,
            nskip=1000,
            ndpost=1000,
            ntree=ntree,
        )
    )


def run_bartz(
    key: Key[Array, ""], train: Data, kwargs: Mapping[str, Any]
) -> Float32[Array, "n_test"]:
    bart = bartz.BART.gbart(train.raw_X, train.y, **kwargs, seed=key)
    pred = bart.yhat_test_mean
    return pred.block_until_ready()


def run_BART3(
    key: Key[Array, ""], train: Data, kwargs: Mapping[str, Any]
) -> Float64[np.ndarray, "n_test"]:
    kw_BART = dict(kwargs)
    kw_BART.update(
        x_test=kwargs["x_test"].T,
        rm_const=False,
        mc_cores=1,
        seed=make_int_seed(key),
    )
    bart = BART3.mc_gbart(train.raw_X.T, train.y, **kw_BART)
    return bart.yhat_test_mean


def run_dbarts(
    key: Key[Array, ""], train: Data, kwargs: Mapping[str, Any]
) -> Float64[np.ndarray, "n_test"]:
    kw_dbarts = dict(kwargs)
    kw_dbarts.update(
        x_test=kwargs["x_test"].T,
        seed=make_int_seed(key),
        keeptrainfits=False,
        keeptrees=False,
    )
    bart = dbarts.bart(train.raw_X.T, train.y, **kw_dbarts)
    return bart.yhat_test_mean


def run_bartMachine(
    key: Key[Array, ""], train: Data, kwargs: Mapping[str, Any]
) -> Float64[np.ndarray, "n_test"]:
    # I can't configure the splitting grid with bartMachine
    kw_bartMachine = dict(kwargs)
    kw_bartMachine.pop("x_test")
    kw_bartMachine.pop("usequants")
    kw_bartMachine.pop("numcut")
    kw_bartMachine.update(
        num_trees=kw_bartMachine.pop("ntree"),
        num_burn_in=kw_bartMachine.pop("nskip"),
        num_iterations_after_burn_in=kw_bartMachine.pop("ndpost"),
        run_in_sample=False,
        sig_sq_est=kw_bartMachine.pop("sigest"),
        # mem_cache_for_speed=False,  # set to False to use less memory
        seed=make_int_seed(key),
    )
    bart = bartMachine.bartMachine(
        pl.DataFrame(np.array(train.raw_X.T)),
        pl.Series(np.array(train.y)),
        **kw_bartMachine,
    )
    return bart.predict(pl.DataFrame(np.array(kwargs["x_test"].T)))


RUNNERS: Mapping[str, Callable] = MappingProxyType(
    dict(
        bartz=run_bartz,
        BART=run_BART3,
        dbarts=run_dbarts,
        bartMachine=run_bartMachine,
    )
)


def run_slave(cfg: Config) -> None:
    """Run a single method on a single dataset and print the results."""
    with pipes(stderr=None, stdout=stderr):
        (n,) = cfg.nvec
        p = cfg.fixed_p
        ntree = cfg.fixed_ntree

        # compute number of runs to do to make sure the "effective sample size"
        # is at least 1000 to reduce the error on the RMSE, but no more than 10
        # runs to avoid overhead
        num = min(10, ceil(1000 / n))

        key = random.key(cfg.seed)
        keys = split(key, 2 * num)

        runner = RUNNERS[cfg.method]
        print(f"run {cfg.method} {num} times...")

        times = []
        mses = []
        for i in range(num):
            print("generate data...")
            train, test = make_split_data(keys.pop(), n, cfg.n_test, p)
            block_until_ready((train, test))
            bartz_kwargs = get_bartz_kwargs(ntree, test)

            print(f"run {cfg.method} ({i + 1}/{num})...")
            with Timer() as timer:
                yhat_test_mean = runner(keys.pop(), train, bartz_kwargs)

            # compute mse and store results
            mse = np.mean(np.square(yhat_test_mean - test.y)).item()
            times.append(timer.time)
            mses.append(mse)

            # free memory
            collect()
            robjects.r("gc()")

        time = np.mean(times).item()
        rmse = np.sqrt(np.mean(mses)).item()
        print(f"{cfg.method} time: {format_time(time)}, rmse: {rmse:.2f}")

        output = dict(
            n=n,
            p=p,
            ntree=ntree,
            prior_var=train.prior_var.item(),
            pop_var=train.pop_var.item(),
            eps_var=train.eps_var.item(),
            time=timer.time,
            rmse=rmse,
        )

    # print output as a json
    print(json.dumps(output))


def run_master(cfg: Config) -> dict[str, dict[str, list[int | float]]]:
    """Run all BART packages on simulated data."""
    # random seed
    key = random.key(cfg.seed)

    # method -> (field -> list of values along n)
    results: dict[str, dict[str, list[int | float]]] = {}

    # list to keep track of which methods timed out
    timed_out: list[str] = []

    for n in cfg.nvec:
        # determine ntree and p for this n
        ntree = (
            max(1, n // cfg.n_over_ntree)
            if cfg.fixed_ntree is None
            else cfg.fixed_ntree
        )
        p = max(1, n // cfg.n_over_p) if cfg.fixed_p is None else cfg.fixed_p
        print(f"\nn = {n:_}, ntree = {ntree:_}, p = {p:_}")

        # split random seed
        keys = split(key, 5)
        key = keys.pop()

        for method in RUNNERS:
            # skip if already timed out previously
            if method in timed_out:
                continue

            # command line to invoke script in slave mode
            cmd = [
                executable,
                __file__,
                "-n",
                str(n),
                "-P",
                str(p),
                "-T",
                str(ntree),
                "-m",
                method,
                "-s",
                str(make_int_seed(keys.pop())),
            ]

            # invoke script in slave mode with timeout
            try:
                proc = run(cmd, stdout=PIPE, text=True, timeout=cfg.timeout)

            # if timed out, continue to next method
            except TimeoutExpired:
                print(f"{method} timed out after {cfg.timeout} seconds")
                timed_out.append(method)
                continue

            # if subprocess failed, crash
            if proc.returncode:
                raise RuntimeError(
                    f"{method} subprocess failed with return code {proc.returncode}"
                )

            # read results from subprocess
            output: dict[str, float | int] = json.loads(proc.stdout)

            # check configs were propagated properly
            assert output["n"] == n
            assert output["p"] == p
            assert output["ntree"] == ntree

            # append results
            result = results.setdefault(method, {})
            for k, v in output.items():
                result.setdefault(k, []).append(v)

        # if all methods timed out, stop benchmarking
        if len(timed_out) == len(RUNNERS):
            print("all methods timed out, stop benchmark")
            break

        # print summary of the results added in this iteration of the loop
        print()
        print()
        for method, result in results.items():
            if method in timed_out:
                continue
            print(
                f"{method:12s}: time = {format_time(result['time'][-1]):>7s}, "
                f"rmse = {result['rmse'][-1]:.2f}"
            )
        print()

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


def parse_args(argv: Sequence[str]) -> Namespace:
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
        "-l",
        "--min-log2-n",
        type=int,
        default=2,
        help="lower end (included) of the n range as log2(n)",
    )
    parser.add_argument(
        "-u",
        "--max-log2-n",
        type=int,
        default=30,
        help="upper end (included) of the n range as log2(n)",
    )
    parser.add_argument(
        "-n",
        "--sample-size",
        metavar="N",
        dest="n",
        type=int,
        default=None,
        help="use fixed sample size N (overrides -u option)",
    )
    parser.add_argument(
        "-P",
        "--num-predictors",
        metavar="P",
        dest="p",
        type=int,
        default=None,
        help="use fixed number of predictors P (overrides -p option)",
    )
    parser.add_argument(
        "-T",
        "--num-trees",
        metavar="T",
        dest="ntree",
        type=int,
        default=None,
        help="use fixed number of trees T (overrides -t option)",
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=list(RUNNERS),
        default=None,
        help="BART method to use",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=2026_01_24_16_54,
        help="random seed",
    )
    args = parser.parse_args(argv)

    # Validate that -n, -P, -T, -m are all specified together or all absent
    nptm_group = [args.n, args.p, args.ntree, args.method]
    nptm_set = [v is not None for v in nptm_group]
    if any(nptm_set) and not all(nptm_set):
        parser.error("-n, -P, -T, and -m must all be specified together or all omitted")

    # If -n, -P, -T, -m are specified, -s is also required
    if all(nptm_set) and args.seed is None:
        parser.error("-s/--seed is required when -n, -P, -T, and -m are specified")

    return args


def args_to_config(args: Namespace) -> Config:
    """Convert command line arguments to a Config object."""
    cfg_kwargs: dict[str, Any] = {}
    if args.high_ntree:
        cfg_kwargs["fixed_ntree"] = None
        cfg_kwargs["n_over_ntree"] = 8
    if args.high_p:
        cfg_kwargs["fixed_p"] = None
        cfg_kwargs["n_over_p"] = 10
    if args.n is None:
        cfg_kwargs["nvec"] = tuple(
            2**p for p in range(args.min_log2_n, args.max_log2_n + 1)
        )
    else:
        cfg_kwargs["nvec"] = (args.n,)
        cfg_kwargs["fixed_p"] = args.p
        cfg_kwargs["fixed_ntree"] = args.ntree
    cfg_kwargs["seed"] = args.seed
    cfg_kwargs["method"] = args.method
    return Config(**cfg_kwargs)


def main(argv: Sequence[str] = argv[1:]):
    """Entry point."""
    args = parse_args(argv)
    cfg = args_to_config(args)
    if args.n is None:
        results = run_master(cfg)
        save_results(results, cfg)
    else:
        run_slave(cfg)


if __name__ == "__main__":
    main()
