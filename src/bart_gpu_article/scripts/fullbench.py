"""Realistic benchmark: fit method, predict on held-out set, compute RMSE."""

import json
import sys
from abc import ABC, abstractmethod
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from collections.abc import Sequence
from contextlib import redirect_stdout
from pathlib import Path
from subprocess import PIPE, TimeoutExpired, run
from time import perf_counter
from typing import Any

import numpy
from bartz import Bart
from bartz.jaxext import get_default_device, split
from equinox import Module
from jax import block_until_ready, config, random
from jax.errors import JaxRuntimeError
from jaxtyping import Array, Float, Key

from bart_gpu_article.datasim import load_data
from bart_gpu_article.scripts.benchmark import (
    EXIT_OUT_OF_MEMORY,
    format_time,
    make_int_seed,
)


class Config(Module):
    """Configuration for the fullbench script."""

    method: str
    platform: str
    slave: bool
    seed: int
    timeout: float
    rounds: int
    test_size: int
    datasets: tuple[str, ...]
    dataset: str | None
    round_seed: int | None

    @property
    def device_kind(self) -> str:
        return get_default_device().device_kind


class Timer:
    """Context manager to time a block of code."""

    time: float

    def __enter__(self) -> Timer:
        self._start = perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.time = perf_counter() - self._start


class Benchmark(ABC):
    """Harness base class: separate `train` and `predict` for per-phase timing."""

    @abstractmethod
    def setup(
        self,
        key: Key[Array, ""],
        x_train: Float[numpy.ndarray, "p n_train"],
        y_train: Float[numpy.ndarray, " n_train"],
        x_test: Float[numpy.ndarray, "p n_test"],
        cfg: Config,
    ) -> None: ...

    @abstractmethod
    def train(self) -> None: ...

    @abstractmethod
    def predict(self) -> Float[numpy.ndarray, " n_test"]: ...

    subclasses: dict[str, type[Benchmark]] = {}

    def __init_subclass__(cls) -> None:
        Benchmark.subclasses[cls.__name__.lower()] = cls


class Bartz(Benchmark):
    """Bartz harness using the high-level `bartz.Bart` interface."""

    def setup(
        self,
        key: Key[Array, ""],
        x_train: Float[numpy.ndarray, "p n_train"],
        y_train: Float[numpy.ndarray, " n_train"],
        x_test: Float[numpy.ndarray, "p n_test"],
        cfg: Config,
    ) -> None:
        self._key = key
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._platform = cfg.platform

    def train(self) -> None:
        self._bart = Bart(
            x_train=self._x_train,
            y_train=self._y_train,
            seed=self._key,
            devices=self._platform,
        )
        block_until_ready(self._bart)

    def predict(self) -> Float[numpy.ndarray, " n_test"]:
        yhat = self._bart.predict(self._x_test, kind="mean")
        return numpy.asarray(yhat)


class Xgboost(Benchmark):
    """Xgboost harness using `XGBRegressor` defaults."""

    def setup(
        self,
        key: Key[Array, ""],
        x_train: Float[numpy.ndarray, "p n_train"],
        y_train: Float[numpy.ndarray, " n_train"],
        x_test: Float[numpy.ndarray, "p n_test"],
        cfg: Config,
    ) -> None:
        from xgboost import XGBRegressor

        self._X_train = x_train.T
        self._y_train = y_train
        self._X_test = x_test.T
        self._model = XGBRegressor(
            random_state=make_int_seed(key),
            device=cfg.platform,
        )

    def train(self) -> None:
        self._model.fit(self._X_train, self._y_train)

    def predict(self) -> Float[numpy.ndarray, " n_test"]:
        return self._model.predict(self._X_test)


EMPTY_ROW_KEYS = (
    "seed",
    "device",
    "method",
    "dataset_path",
    "n_train",
    "n_test",
    "time_train",
    "time_test",
    "rmse",
)


def run_slave(cfg: Config) -> dict[str, Any]:
    """Run one (dataset, seed) unit and return its row as a dict."""
    assert cfg.dataset is not None
    assert cfg.round_seed is not None

    print(f"load dataset {cfg.dataset}...")
    data = load_data(cfg.dataset)
    data = block_until_ready(data)

    if data.raw_X is None:
        raise RuntimeError(
            f"dataset {cfg.dataset} has no raw_X; both bartz and xgboost need it"
        )

    raw_X = numpy.asarray(data.raw_X)
    y = numpy.asarray(data.y)
    del data

    n_total = int(y.size)
    n_test = cfg.test_size
    n_train = n_total - n_test
    if n_train <= 0:
        raise RuntimeError(f"n_total={n_total} <= test_size={n_test}")
    print(f"n_train={n_train:_}, n_test={n_test:_}")

    keys = split(random.key(cfg.round_seed))

    perm = numpy.asarray(random.permutation(keys.pop(), n_total))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    x_train = raw_X[:, train_idx]
    y_train = y[train_idx]
    x_test = raw_X[:, test_idx]
    y_test = y[test_idx]

    del raw_X, y

    print(f"setup {cfg.method}...")
    bench = Benchmark.subclasses[cfg.method]()
    bench.setup(keys.pop(), x_train, y_train, x_test, cfg)
    del x_train, y_train, x_test

    print("train...")
    with Timer() as t_train:
        bench.train()
    print(f"train time: {format_time(t_train.time)}")

    print("predict...")
    with Timer() as t_test:
        yhat = bench.predict()
    print(f"predict time: {format_time(t_test.time)}")

    rmse = numpy.sqrt(numpy.mean(numpy.square(yhat - y_test))).item()
    print(f"rmse: {rmse:.4f}")

    return dict(
        seed=cfg.round_seed,
        device=cfg.device_kind,
        method=cfg.method,
        dataset_path=cfg.dataset,
        n_train=n_train,
        n_test=n_test,
        time_train=t_train.time,
        time_test=t_test.time,
        rmse=rmse,
    )


def slave_loop(cfg: Config) -> dict[str, Any]:
    """Run `run_slave` with OOM handling, exit with code on OOM."""
    try:
        return run_slave(cfg)
    except JaxRuntimeError as exc:
        if not exc.args[0].startswith(
            "RESOURCE_EXHAUSTED: Out of memory while trying to allocate"
        ):
            raise
        print(f"\nOut-of-memory:\n{exc}")
        sys.exit(EXIT_OUT_OF_MEMORY)


def _null_row(method: str, dataset: str, seed: int, device_kind: str) -> dict[str, Any]:
    return dict(
        seed=seed,
        device=device_kind,
        method=method,
        dataset_path=dataset,
        n_train=None,
        n_test=None,
        time_train=None,
        time_test=None,
        rmse=None,
    )


def run_master(cfg: Config) -> dict[str, list]:
    """Loop over datasets and rounds, spawning slave subprocesses."""
    print(
        f"\nfullbench {cfg.method} on {len(cfg.datasets)} dataset(s)"
        f" x {cfg.rounds} round(s)..."
    )
    columns: dict[str, list] = {k: [] for k in EMPTY_ROW_KEYS}
    master_key = random.key(cfg.seed)

    for ds_path in cfg.datasets:
        for r in range(cfg.rounds):
            keys = split(master_key)
            master_key = keys.pop()
            round_seed = make_int_seed(keys.pop())

            cmd = [
                sys.executable,
                __file__,
                "--slave",
                "-m",
                cfg.method,
                "-d",
                cfg.platform,
                "--test-size",
                str(cfg.test_size),
                "--dataset",
                ds_path,
                "--round-seed",
                str(round_seed),
            ]

            print(
                f"\n=== dataset={ds_path} round={r + 1}/{cfg.rounds}"
                f" seed={round_seed} ==="
            )

            row: dict[str, Any]
            try:
                proc = run(cmd, stdout=PIPE, text=True, timeout=cfg.timeout)
            except TimeoutExpired:
                print(f"timeout after {cfg.timeout}s")
                row = _null_row(cfg.method, ds_path, round_seed, cfg.device_kind)
            else:
                if proc.returncode == 0:
                    row = json.loads(proc.stdout.strip())
                elif proc.returncode == EXIT_OUT_OF_MEMORY:
                    print("out of memory")
                    row = _null_row(cfg.method, ds_path, round_seed, cfg.device_kind)
                else:
                    raise RuntimeError(
                        f"slave exited with code {proc.returncode};"
                        f" stdout: {proc.stdout!r}"
                    )

            for k in EMPTY_ROW_KEYS:
                columns[k].append(row[k])

    return columns


def output_path(cfg: Config) -> Path:
    """Path of the master's results file."""
    device_kind = cfg.device_kind.replace(" ", "_")
    return Path("./results") / f"fullbench-{cfg.method}-{device_kind}.json"


def save_results(cfg: Config, results: dict) -> None:
    """Write slave row to stdout or master table to disk."""
    if cfg.slave:
        assert isinstance(results, dict)
        print(json.dumps(results))
        return

    assert isinstance(results, dict)
    path = output_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"write {path}...")
    with open(path, "w") as f:
        json.dump(results, f, indent=4)


def parse_args(argv: Sequence[str]) -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=list(Benchmark.subclasses),
        required=True,
        help="regression method to benchmark",
    )
    parser.add_argument(
        "-d",
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="device to run the method on",
    )
    parser.add_argument(
        "-r",
        "--rounds",
        type=int,
        default=5,
        help="number of random train/test splits per dataset (master mode)",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=1000,
        help="fixed size of the held-out test set",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=2026_04_24_12_00,
        help="master random seed",
    )
    parser.add_argument(
        "-T",
        "--timeout",
        type=float,
        default=3600,
        help="per-slave timeout in seconds",
    )
    parser.add_argument(
        "--slave",
        action="store_true",
        help="run one (dataset, seed) unit and print a JSON row on stdout",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="dataset path (slave mode)",
    )
    parser.add_argument(
        "--round-seed",
        type=int,
        default=None,
        help="integer seed passed to the slave (slave mode)",
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="dataset paths (master mode)",
    )
    args = parser.parse_args(argv)

    if args.slave:
        if args.dataset is None or args.round_seed is None:
            parser.error("--slave requires --dataset and --round-seed")
        if args.datasets:
            parser.error("positional datasets not allowed with --slave")
    else:
        if not args.datasets:
            parser.error("at least one dataset is required in master mode")
        if args.dataset is not None or args.round_seed is not None:
            parser.error("--dataset and --round-seed are slave-only (use --slave)")

    return args


def args_to_config(args: Namespace) -> Config:
    """Convert command line arguments to a Config."""
    return Config(
        method=args.method,
        platform=args.device,
        slave=args.slave,
        seed=args.seed,
        timeout=args.timeout,
        rounds=args.rounds,
        test_size=args.test_size,
        datasets=tuple(args.datasets),
        dataset=args.dataset,
        round_seed=args.round_seed,
    )


def setup_device(cfg: Config) -> None:
    """Configure the jax device."""
    match cfg.platform:
        case "cpu":
            # disable gpu altogether, and create multiple cpu devices
            config.update("jax_platforms", "cpu")
            config.update("jax_num_cpu_devices", 4)
            # 4 cpu devices because bartz uses 4 chains by default
        case "gpu":
            # jax would do the same, but by setting it explicitly, we are
            # forcing an error if there's no gpu
            config.update("jax_platforms", "cuda,cpu")
        case _:
            raise ValueError(cfg.platform)


def main(argv: Sequence[str] = sys.argv[1:]) -> None:
    """Entry point."""
    with redirect_stdout(sys.stderr):
        args = parse_args(argv)
        cfg = args_to_config(args)

        setup_device(cfg)

        if not cfg.slave:
            path = output_path(cfg)
            if path.exists():
                print(f"output file {path} already exists, terminating")
                return

        if cfg.slave:
            results = slave_loop(cfg)
        else:
            results = run_master(cfg)

    save_results(cfg, results)


if __name__ == "__main__":
    main()
