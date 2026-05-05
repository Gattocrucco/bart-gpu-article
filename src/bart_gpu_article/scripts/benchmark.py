"""Speed benchmark of bartz & competitors."""

import json
import math
import platform
import sys
from abc import ABC, abstractmethod
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from collections.abc import Callable, Sequence
from contextlib import redirect_stdout
from functools import partial
from gc import collect
from pathlib import Path
from subprocess import PIPE, TimeoutExpired, run
from time import perf_counter
from typing import Any, Literal

import numpy
from bartz.jaxext import split
from bartz.mcmcloop import run_mcmc
from bartz.mcmcstep import State, init, make_p_nonterminal
from equinox import Module
from jax import (
    Device,
    block_until_ready,
    config,
    debug,
    default_device,
    device_put,
    devices,
    jit,
    random,
)
from jax import numpy as jnp
from jax.errors import JaxRuntimeError
from jaxtyping import Array, Key

from bart_gpu_article.datasim import Data, make_data


class UnitConfig(Module):
    """Configuration for a single benchmark unit."""

    n: int
    ntree: int
    p: int
    reps: int
    steps_per_rep: int
    device: Device
    data_device: Device
    benchclass: type["Benchmark"]
    quantize_x: bool


class Config(Module):
    """General configuration of the script."""

    platform: Literal["cpu", "gpu"]
    benchlabel: str
    nvec: tuple[int, ...]
    slave: bool
    seed: int
    timeout: float
    fixed_ntree: int | None = 200
    fixed_p: int | None = 100
    n_over_ntree: int | None = None
    n_over_p: int | None = None
    reps: int = 3

    def device(self) -> Device:
        """Get the jax device to use for running the algorithm."""
        return devices(self.platform)[0]

    def data_device(self) -> Device:
        """Get the jax device to use for data generation."""
        platform = "cpu" if self.benchlabel == "xgboost" else self.platform
        return devices(platform)[0]

    def quantize_x(self) -> bool:
        """Determine whether X should be quantized or original."""
        return self.benchlabel == "bartz"

    def unit_config(self, n: int) -> UnitConfig:
        """Return the specific config at sample size `n`."""
        return UnitConfig(
            n=n,
            ntree=max(1, n // self.n_over_ntree)
            if self.fixed_ntree is None
            else self.fixed_ntree,
            p=max(1, n // self.n_over_p) if self.fixed_p is None else self.fixed_p,
            reps=self.reps,
            steps_per_rep=1 if self.benchlabel == "xgboost" else 15,
            device=self.device(),
            data_device=self.data_device(),
            benchclass=Benchmark.subclasses[self.benchlabel],
            quantize_x=self.quantize_x(),
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


def format_mem(m: float) -> str:
    """Format an amount of memory in bytes."""
    return f"{num2si(m)}B"


def cpu_brand() -> str:
    """Return a human-readable CPU model string (e.g. 'Apple M1 Pro')."""
    system = platform.system()
    if system == "Darwin":
        proc = run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stdout=PIPE,
            text=True,
            check=True,
        )
        brand = proc.stdout.strip()
        if brand:
            return brand
    elif system == "Linux":
        # /proc/cpuinfo "model name" works on x86; on aarch64 it's missing and
        # we fall back to the SoC string from "Hardware" or "CPU implementer"
        try:
            with open("/proc/cpuinfo") as f:
                cpuinfo = f.read()
        except OSError:
            cpuinfo = ""
        for key in ("model name", "Hardware", "cpu model"):
            for line in cpuinfo.splitlines():
                if line.startswith(key):
                    _, _, value = line.partition(":")
                    value = value.strip()
                    if value:
                        return value
    return platform.processor() or platform.machine() or "cpu"


def device_kind(device: Device) -> str:
    """Return the device identifier used in result filenames and JSON output."""
    if device.platform == "cpu":
        return cpu_brand()
    return device.device_kind


class Stop(Exception):
    """Exception to be raised to stop the benchmarking loop."""


class Benchmark(ABC):
    """Base class for benchmark harnesses."""

    @abstractmethod
    def setup(self, key: Key[Array, ""], data: Data, cfg: UnitConfig) -> None:
        """Set up the benchmark initial state."""
        ...

    def setup_run(self) -> None:
        """Method run before each timed `run`, but not timed, default does nothing."""
        pass

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
        expected_memory_usage = cfg.n * (cfg.ntree + cfg.p)
        print(f"expected memory usage: {format_mem(expected_memory_usage)}")

        print("initialize mcmc state...")
        assert cfg.data_device == cfg.device
        with default_device(cfg.device):
            # init creates new arrays on the default device
            self.state = init(
                X=data.quantized_X,
                y=data.y,
                offset=0.0,
                max_split=data.max_split,
                num_trees=cfg.ntree,
                # 6 is bartz.Bart default, note the convention is different than
                # xgboost, which at its default "maxdepth=6" will have one level
                # more
                p_nonterminal=make_p_nonterminal(6, 0.95, 2),
                leaf_prior_cov_inv=jnp.float32(cfg.ntree),
                error_cov_df=2.0,
                error_cov_scale=2.0,
                min_points_per_decision_node=10 if cfg.n > 10 else None,
            )
        del data
        key = device_put(key, cfg.device)
        self.device = cfg.device

        print("compile mcmc loop...")

        @partial(jit, donate_argnums=(1,))
        def run_bart(key: Key[Array, ""], bart: State) -> State:
            def callback(**_):
                return debug.callback(lambda: print(".", end="", flush=True))

            bart, _, _ = run_mcmc(key, bart, cfg.steps_per_rep, callback=callback)
            return bart

        self.run_bart = run_bart.lower(key, self.state).compile()

        # make sure the state is ready before timing
        self.state = block_until_ready(self.state)

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

        expected_memory_usage = 24 * cfg.n * (cfg.ntree + cfg.p)
        print(f"expected memory usage: {format_mem(expected_memory_usage)}")

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
        """Build the xgboost training matrix and stash params for `run`.

        Quantile binning (the X-quantization equivalent to what bartz does at
        setup) happens here at QuantileDMatrix construction. The booster
        itself is (re)created in `setup_run` so each rep starts from a fresh
        state: unlike bartz/dbarts (which are stateful MCMC and meant to keep
        stepping), xgboost's "iteration" is a fresh ntree-tree fit, and
        keeping a single booster across reps would let residuals drift and
        the per-rep work change.
        """
        import xgboost as xgb

        # one xgboost "iteration" is defined as one ntree-tree fit, so the
        # `run` loop does cfg.ntree boost rounds and steps_per_rep must be 1
        assert cfg.steps_per_rep == 1, cfg.steps_per_rep

        print(f"n * p = {cfg.n * cfg.p:_}, n * ntree = {cfg.n * cfg.ntree:_}")

        # convert to numpy (xgboost expects (n, p) shape) just because you
        # never know
        X = numpy.array(data.raw_X.T)
        y = numpy.array(data.y)

        print("build QuantileDMatrix (quantile binning)...")
        self.dtrain = xgb.QuantileDMatrix(X, label=y)

        # objective is the XGBRegressor default. nthread is omitted so xgboost
        # picks the maximum number of available threads.
        self.params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "device": cfg.device.platform,
            "seed": make_int_seed(key),
            "verbosity": 2,
        }
        self.num_boost_round = cfg.ntree

    def setup_run(self) -> None:
        """Build a fresh booster so each timed `run` starts from the same state."""
        import xgboost as xgb

        self.booster = xgb.Booster(self.params, [self.dtrain])

    def run(self, key: Key[Array, ""]) -> None:
        """Run one full set of ntree boost rounds."""
        for i in range(self.num_boost_round):
            self.booster.update(self.dtrain, i)


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
    with default_device(cfg.data_device):
        data = make_data(keys.pop(), cfg.n, cfg.p, quantized_x=cfg.quantize_x)
        block_until_ready(data)

    # the harness prints its own messages
    bench = cfg.benchclass()
    bench.setup(keys.pop(), data, cfg)
    del data

    print("run...")
    times = []
    for i in range(cfg.reps):
        bench.setup_run()
        print(f"run {i + 1}/{cfg.reps} ", end="", flush=True)
        time = clock(bench.run, keys.pop())
        print(
            f" {cfg.steps_per_rep} iterations in {format_time(time)} ({format_time(time / cfg.steps_per_rep)} per iteration)"
        )
        times.append(time)

    bench.teardown()

    return min(times) / cfg.steps_per_rep


def benchmark_loop(config: Config) -> Any:
    """Run all benchmark units."""
    if config.slave:
        return benchmark_loop_slave(config)
    else:
        return benchmark_loop_master(config)


EXIT_OUT_OF_MEMORY = 17
EXIT_STOP_REQUESTED = 31


def benchmark_loop_slave(config: Config) -> float:
    """Run a single benchmark unit and return the time per iteration."""

    (n,) = config.nvec
    key = random.key(config.seed)

    try:
        # run benchmark unit
        time_per_iter = benchmark_unit(key, config.unit_config(n))

    except JaxRuntimeError as exc:
        if "Out of memory" not in str(exc):
            # unknown error, don't catch
            raise
        else:
            # out of memory, stop the loop
            print(f"\nStop benchmark loop with out-of-memory error:\n{exc}")
            sys.exit(EXIT_OUT_OF_MEMORY)

    except Stop as exc:
        # stop the loop
        print(f"\nStop benchmark loop with exception:\n{exc}")
        sys.exit(EXIT_STOP_REQUESTED)

    else:
        # all normal, return result
        return time_per_iter


def benchmark_loop_master(config: Config) -> dict[str, list]:
    """Run all benchmark units."""

    print(f"\nbenchmark {config.benchlabel}...")

    results: dict[str, list] = {"n": [], "time_per_iter": []}
    key = random.key(config.seed)

    for n in config.nvec:
        # split random key
        keys = split(key)
        key = keys.pop()

        # build command line arguments for slave subprocess
        cmd = [
            sys.executable,
            __file__,
            "-n",
            str(n),
            "-m",
            config.benchlabel,
            "-d",
            config.platform,
            "-s",
            str(make_int_seed(keys.pop())),
        ]
        if config.fixed_ntree is None:
            cmd.append("-t")
        if config.fixed_p is None:
            cmd.append("-p")

        # run subprocess and capture output
        try:
            proc = run(cmd, stdout=PIPE, text=True, timeout=config.timeout)
        except TimeoutExpired:
            print(
                f"\nStop benchmark loop due to timeout after {config.timeout} seconds"
            )
            break

        # check exit status and handle accordingly
        if proc.returncode == 0:
            # parse output and collect results
            time_per_iter = float(proc.stdout.strip())
            results["n"].append(n)
            results["time_per_iter"].append(time_per_iter)

        elif proc.returncode in (EXIT_OUT_OF_MEMORY, EXIT_STOP_REQUESTED):
            # out of memory or stop requested, break the loop
            break

        else:
            # unrecognized error, raise exception
            raise RuntimeError(
                f"Subprocess exited with code {proc.returncode}, stdout: {proc.stdout}"
            )

    return results


def save_results(cfg: Config, results: Any) -> None:
    """Save results in machine-readable format."""
    # in slave mode, print to stdout and return
    if cfg.slave:
        assert isinstance(results, float)
        print(results)
        return

    # write all output in a dictionary
    output = {
        "package": cfg.benchlabel,
        "device_kind": device_kind(cfg.device()),
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
    suffix = f"-{output['package']}-{output['device_kind'].replace(' ', '_')}"
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
            # force an error if gpu not found, and set cpu as default such that
            # data is generated on cpu and when using xgboost jax won't squat
            # the gpu memory
            config.update("jax_platforms", "cpu,cuda")
        case _:
            raise ValueError(cfg.platform)


def parse_args(argv: Sequence[str]) -> Namespace:
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
        "-l",
        "--min-log2-n",
        type=int,
        default=1,
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
        "-d",
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="device to run the benchmark on",
    )
    parser.add_argument(
        "-n",
        "--sample-size",
        type=int,
        default=None,
        metavar="N",
        dest="n",
        help="single n value to benchmark (overrides -l and -u)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=2026_01_24_16_54,
        help="random seed for data generation",
    )
    parser.add_argument(
        "-T",
        "--timeout",
        type=float,
        default=120,
        help="timeout in seconds",
    )
    return parser.parse_args(argv)


def args_to_config(args: Namespace) -> Config:
    """Convert command line arguments to a Config object."""
    cfg_kwargs: dict[str, Any] = {"benchlabel": args.method}
    if args.high_ntree:
        cfg_kwargs["fixed_ntree"] = None
        cfg_kwargs["n_over_ntree"] = 8
    if args.high_p:
        cfg_kwargs["fixed_p"] = None
        cfg_kwargs["n_over_p"] = 10
    cfg_kwargs["slave"] = args.n is not None
    if args.n is not None:
        cfg_kwargs["nvec"] = (args.n,)
    else:
        cfg_kwargs["nvec"] = tuple(
            2**p for p in range(args.min_log2_n, args.max_log2_n + 1)
        )
    cfg_kwargs["platform"] = args.device
    cfg_kwargs["seed"] = args.seed
    cfg_kwargs["timeout"] = args.timeout
    return Config(**cfg_kwargs)


def main(argv: Sequence[str] = sys.argv[1:]) -> None:
    """Entry point of the script."""
    with redirect_stdout(sys.stderr):
        args = parse_args(argv)
        cfg = args_to_config(args)
        setup_device(cfg)
        results = benchmark_loop(cfg)

    # save results may write to stdout
    save_results(cfg, results)


if __name__ == "__main__":
    main()
