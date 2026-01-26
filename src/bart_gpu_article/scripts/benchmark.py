"""Speed benchmark of bartz & competitors."""

import json
import math
from abc import ABC, abstractmethod
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from collections.abc import Callable
from functools import partial
from gc import collect
from os import putenv
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from bartz import mcmcloop
from bartz.jaxext import split
from bartz.mcmcstep import State, init, make_p_nonterminal
from bartz.prepcovars import bin_predictors, quantilized_splits_from_matrix
from bartz.testing import gen_data
from equinox import Module
from jax import (
    Device,
    block_until_ready,
    config,
    debug,
    device_put,
    devices,
    jit,
    random,
)
from jax import numpy as jnp
from jax.errors import JaxRuntimeError
from jaxtyping import Array, Float, Float32, Key, UInt


class UnitConfig(Module):
    """Configuration for a single benchmark unit."""

    n: int
    ntree: int
    p: int
    bartz_xgboost_maxdepth: int
    reps: int
    steps_per_rep: int
    cpu_max_memory: int
    xgboost_gpu_max_n_times_p: int
    device: Device
    benchclass: type["Benchmark"]


class Config(Module):
    """General configuration of the script."""

    platform: Literal["cpu", "gpu"]
    benchlabel: str
    nvec: tuple[int, ...]
    fixed_ntree: int | None = 200
    fixed_p: int | None = 100
    n_over_ntree: int | None = None
    n_over_p: int | None = None
    bartz_xgboost_maxdepth: int = 6
    reps: int = 2
    steps_per_rep: int = 15
    cpu_max_memory: int = 16 * 2**30
    xgboost_gpu_max_n_times_p: int = 2**32
    seed: int = 2026_01_24_16_54

    def device(self) -> Device:
        """Get the jax device to use."""
        return devices(self.platform)[0]

    def unit_config(self, n: int) -> UnitConfig:
        """Return the specific config at sample size `n`."""
        return UnitConfig(
            n=n,
            ntree=max(1, n // self.n_over_ntree)
            if self.fixed_ntree is None
            else self.fixed_ntree,
            p=max(1, n // self.n_over_p) if self.fixed_p is None else self.fixed_p,
            bartz_xgboost_maxdepth=self.bartz_xgboost_maxdepth,
            reps=self.reps,
            steps_per_rep=self.steps_per_rep,
            cpu_max_memory=self.cpu_max_memory,
            xgboost_gpu_max_n_times_p=self.xgboost_gpu_max_n_times_p,
            device=self.device(),
            benchclass=Benchmark.subclasses[self.benchlabel],
        )


class Data(Module):
    """Simulated data."""

    raw_X: Float[Array, "p n"]
    quantized_X: UInt[Array, "p n"]
    y: Float32[Array, " n"]
    max_split: UInt[Array, " p"]
    prior_var: Float32[Array, ""]
    pop_var: Float32[Array, ""]
    eps_var: Float32[Array, ""]


@partial(jit, static_argnums=(1, 2))
def make_data(key: Key[Array, ""], n: int, p: int) -> Data:
    """Generate data."""
    # generate data
    sigma2 = 1 / 3
    data = gen_data(
        key,
        n=n,
        p=p,
        k=1,
        q=2 if p > 2 else 0,
        sigma2_lin=sigma2,
        sigma2_quad=sigma2,
        sigma2_eps=sigma2,
        lam=0.0,
    )

    # quantize predictors
    splits, max_split = quantilized_splits_from_matrix(data.x, 255)
    X = bin_predictors(data.x, splits)

    # squeeze away multivariate outcome
    y = data.y.squeeze(0)

    return Data(
        raw_X=data.x,
        quantized_X=X,
        y=y,
        max_split=max_split,
        prior_var=data.sigma2_pri,
        pop_var=data.sigma2_pop,
        eps_var=data.sigma2_eps,
    )


def num2si(
    x: float,
    fmt: Callable[[float], str] = lambda x: f"{x:#.3g}".rstrip("."),
    si: bool = True,
    space: str = " ",
) -> str:
    """Format a number using SI prefixes."""
    if x == 0:
        return fmt(x) + space
    exp = int(math.floor(math.log10(abs(x))))
    exp3 = exp - (exp % 3)
    x3 = x / (10**exp3)

    if si and exp3 >= -24 and exp3 <= 24 and exp3 != 0:
        exp3_text = space + "yzafpnμm kMGTPEZY"[(exp3 - (-24)) // 3]
    elif exp3 == 0:
        exp3_text = space
    else:
        exp3_text = f"e{exp3}"

    return f"{fmt(x3)}{exp3_text}"


def format_time(t: float) -> str:
    """Format a time as multiple of seconds."""
    return f"{num2si(t)}s"


class Skip(Exception):
    """Exception to be raised to skip a benchmark unit."""


class Benchmark(ABC):
    """Base class for benchmark harnesses."""

    @abstractmethod
    def setup(self, key: Key[Array, ""], data: Data, cfg: UnitConfig) -> None:
        """Set up the benchmark initial state."""
        ...

    @abstractmethod
    def run(self, key: Key[Array, ""]) -> None:
        """Run the thing being benchmarked, may update internal state."""
        ...

    def teardown(self) -> None:
        """Method run at the end of the benchmark unit, default does nothing."""
        pass

    subclasses: dict[str, type[Benchmark]] = {}

    def __init_subclass__(cls):
        """Add subclasses to `Benchmark.subclasses`."""
        Benchmark.subclasses[cls.__name__.lower()] = cls


class Bartz(Benchmark):
    """Benchmark harness for the bartz mcmc step."""

    def setup(self, key: Key[Array, ""], data: Data, cfg: UnitConfig) -> None:
        """Create the initial bart state and compile the mcmc loop."""
        # decide whether to skip
        expected_memory_usage = cfg.n * (cfg.ntree + cfg.p)
        if cfg.device.platform == "cpu" and expected_memory_usage > cfg.cpu_max_memory:
            # on cpu, jax won't raise out of memory errors, and just hang forever
            raise Skip
        print(f"expected memory usage: {num2si(expected_memory_usage)}B")

        print("initialize mcmc state...")
        # we first put all arguments on the device and only afterwards call init
        # because init uses the arguments to determine the device we are working
        # on and automatically configure settings related to performance.
        kwargs = dict(
            X=data.quantized_X,
            y=data.y,
            offset=0.0,
            max_split=data.max_split,
            num_trees=cfg.ntree,
            p_nonterminal=make_p_nonterminal(cfg.bartz_xgboost_maxdepth, 0.95, 2),
            leaf_prior_cov_inv=jnp.float32(cfg.ntree),
            error_cov_df=2.0,
            error_cov_scale=2.0,
            min_points_per_decision_node=10 if cfg.n > 10 else None,
        )
        key, kwargs = device_put((key, kwargs), cfg.device, donate=True)
        self.state = init(**kwargs)
        self.device = cfg.device

        print("compile mcmc loop...")

        @partial(jit, donate_argnums=(1,))
        def run_bart(key: Key[Array, ""], bart: State) -> State:
            def callback(**_):
                return debug.callback(lambda: print(".", end="", flush=True))

            bart, _, _ = mcmcloop.run_mcmc(
                key, bart, cfg.steps_per_rep, callback=callback
            )
            return bart

        self.run_bart = run_bart.lower(key, self.state).compile()

    def run(self, key: Key[Array, ""]) -> None:
        """Run a few iterations of the mcmc and update the state."""
        key = device_put(key, self.device)
        self.state = block_until_ready(self.run_bart(key, self.state))


def make_int_seed(key: Key[Array, ""]) -> int:
    """Convert a jax random key to a positive integer that fits into int32."""
    return random.randint(key, (), 0, jnp.uint32(2**31), jnp.uint32).item()


class Dbarts(Benchmark):
    """Benchmark harness for the dbarts mcmc step."""

    def setup(self, key: Key[Array, ""], data: Data, cfg: UnitConfig) -> None:
        """Create the initial dbarts state."""
        from bart_gpu_article.rbartpackages.dbarts import dbarts, dbartsControl

        # check device
        if cfg.device.platform != "cpu":
            raise RuntimeError("dbarts only works on cpu")

        # decide whether to skip
        expected_memory_usage = 24 * cfg.n * (cfg.ntree + cfg.p)
        if expected_memory_usage > cfg.cpu_max_memory:
            raise Skip
        print(f"expected memory usage: {num2si(expected_memory_usage)}B")

        print("initialize dbarts state...")
        control = dbartsControl(
            verbose=True,
            keepTrainingFits=False,
            keepTrees=False,
            n_cuts=255,
            n_trees=cfg.ntree,
            n_chains=1,
            n_threads=1,
            printEvery=1,
            rngSeed=make_int_seed(key),
        )
        self.sampler = dbarts(
            data.raw_X.T, data.y, control=control, sigma=2 * data.eps_var.item() ** 0.5
        )
        self.ndpost = cfg.steps_per_rep

    def run(self, key: Key[Array, ""]) -> None:
        """Run the dbarts mcmc."""
        self.sampler.run(0, self.ndpost)

    def teardown(self) -> None:
        """Clean up R memory."""
        from rpy2 import robjects

        del self.sampler
        collect()
        robjects.r("gc()")


class Xgboost(Benchmark):
    """Benchmark harness for xgboost."""

    def setup(self, key: Key[Array, ""], data: Data, cfg: UnitConfig) -> None:
        """Create the xgboost model."""
        from xgboost import XGBRegressor

        # decide whether to skip based on memory/time limits
        if cfg.device.platform == "cpu":
            max_n_times_ntree = 2**32  # time limit
            if (
                cfg.n * cfg.p > cfg.cpu_max_memory // 16
                or cfg.n * cfg.ntree > max_n_times_ntree
            ):
                raise Skip
        else:  # gpu
            # to avoid out-of-memory session termination
            if cfg.n * cfg.p > cfg.xgboost_gpu_max_n_times_p:
                raise Skip

        print(f"n * p = {cfg.n * cfg.p:_}, n * ntree = {cfg.n * cfg.ntree:_}")

        # store data for fitting (xgboost expects (n, p) shape)
        self.X = data.raw_X.T
        self.y = data.y

        print("define xgboost model...")
        self.model = XGBRegressor(
            n_estimators=cfg.ntree,
            max_depth=cfg.bartz_xgboost_maxdepth - 1,
            n_jobs=1,
            random_state=make_int_seed(key),
            device=cfg.device.platform,
            verbosity=2,
        )

    def run(self, key: Key[Array, ""]) -> None:
        """Fit the xgboost model."""
        self.model.fit(self.X, self.y, verbose=True)


def clock(f: Callable, *args: Any) -> float:
    """Time a function call."""
    start = perf_counter()
    f(*args)
    end = perf_counter()
    return end - start


def benchmark_unit(key: Key[Array, ""], cfg: UnitConfig) -> float:
    """Run a single benchmark unit."""
    # print information
    print(f"\nn = {cfg.n:_}, ntree = {cfg.ntree:_}, p = {cfg.p:_}")

    # split random seed
    keys = list(random.split(key, cfg.reps + 3))
    key = keys.pop()

    print("generate data...")
    data = make_data(keys.pop(), cfg.n, cfg.p)

    # the harness prints its own messages
    bench = cfg.benchclass()
    bench.setup(keys.pop(), data, cfg)

    print("run...")
    times = []
    for i in range(cfg.reps):
        print(f"run {i + 1}/{cfg.reps} ", end="", flush=True)
        time = clock(bench.run, keys.pop())
        print(
            f" {cfg.steps_per_rep} iterations in {format_time(time)} ({format_time(time / cfg.steps_per_rep)} per iteration)"
        )
        times.append(time)

    bench.teardown()

    return min(times) / cfg.steps_per_rep


def benchmark_loop(config: Config) -> dict[str, list]:
    """Run all benchmark units."""
    key = random.key(config.seed)

    print(f"\nbenchmark {config.benchlabel}...")

    results = {}
    for n in config.nvec:
        # split random key
        keys = split(key)
        key = keys.pop()

        try:
            # run benchmark unit
            time_per_iter = benchmark_unit(keys.pop(), config.unit_config(n))

        except JaxRuntimeError as exc:
            # suppress out-of-memory errors, without saving results
            if not exc.args[0].startswith(
                "RESOURCE_EXHAUSTED: Out of memory while trying to allocate"
            ):
                raise

        except Skip:
            # don't save results
            pass

        else:
            # save results
            results.setdefault("n", []).append(n)
            results.setdefault("time_per_iter", []).append(time_per_iter)

        # free memory
        collect()

    return results


def save_results(cfg: Config, results: dict[str, list[Any]]) -> None:
    """Save results in machine-readable format."""
    # write all output in a dictionary
    output = {
        "package": cfg.benchlabel,
        "device_kind": cfg.device().device_kind,
        "maxdepth": cfg.bartz_xgboost_maxdepth,
        "results": results,
    }
    if cfg.fixed_ntree is None:
        output["n/ntree"] = cfg.n_over_ntree
    else:
        output["ntree"] = cfg.fixed_ntree
    if cfg.fixed_p is None:
        output["n/p"] = cfg.n_over_p
    else:
        output["p"] = cfg.fixed_p

    # determine filename suffix
    suffix = f"-{output['package']}-{output['device_kind']}"
    if "n/ntree" in output:
        suffix += "-highntree"
    if "n/p" in output:
        suffix += "-highp"

    # save dictionary with output as json file
    results_dir = Path("./results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"benchmark{suffix}.json"
    print(f"write {output_path}...")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)


def setup_device(cfg: Config) -> None:
    """Configure the jax device."""
    match cfg.platform:
        case "cpu":
            # disable gpu altogether
            config.update("jax_platforms", "cpu")
        case "gpu":
            if cfg.benchlabel == "bartz":
                # allocate all gpu memory
                putenv("XLA_PYTHON_CLIENT_MEM_FRACTION", ".99")
            else:
                # do not let jax actually use the gpu
                putenv("XLA_PYTHON_CLIENT_MEM_FRACTION", ".00")

            # force an error if gpu not found
            config.update("jax_platforms", "cuda,cpu")
        case _:
            raise ValueError(cfg.platform)


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=Benchmark.subclasses.keys(),
        default="bartz",
        help="which regression method to benchmark",
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
        default=30,
        help="upper end (included) of the n range as log2(n)",
    )
    parser.add_argument(
        "-d",
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="device to run the benchmark on",
    )
    return parser.parse_args()


def args_to_config(args: Namespace) -> Config:
    """Convert command line arguments to a Config object."""
    cfg_kwargs: dict[str, Any] = {"benchlabel": args.method}
    if args.high_ntree:
        cfg_kwargs["fixed_ntree"] = None
        cfg_kwargs["n_over_ntree"] = 8
    if args.high_p:
        cfg_kwargs["fixed_p"] = None
        cfg_kwargs["n_over_p"] = 10
    cfg_kwargs["nvec"] = tuple(2**p for p in range(1, args.max_log2_n + 1))
    cfg_kwargs["platform"] = args.device
    return Config(**cfg_kwargs)


def main() -> None:
    """Entry point of the script."""
    args = parse_args()
    cfg = args_to_config(args)
    setup_device(cfg)
    results = benchmark_loop(cfg)
    save_results(cfg, results)


if __name__ == "__main__":
    main()
