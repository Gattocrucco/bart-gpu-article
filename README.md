[![DOI](https://zenodo.org/badge/880171976.svg)](https://doi.org/10.5281/zenodo.14010504)

# bart-gpu-article

Code to reproduce the results in Petrillo (2024), "Very fast Bayesian Additive Regression Trees on GPU", [arXiv:2410.23244](https://arxiv.org/abs/2410.23244).


## Hardware setup

Running the GPU benchmarks requires a NVIDIA GPU. Virtual machines with GPUs can be conveniently rented in places like https://cloud.vast.ai starting from about 0.10 $/hour.

Other things can be run on any computer with at least 16 GB or RAM.


## Files setup

* Copy/clone the files to your computer
* Set the working directory to `bart-gpu-article`


## R setup

R is needed only to run benchmarks involving R packages, and in particular java is needed only for `bartMachine`, which is used only in the `test-rmse` command.

Install:

  * R 4.5.1 https://www.r-project.org
  
  * JDK 25.0.2 https://jdk.java.net (on macOS, put the directory into `/Library/Java/JavaVirtualMachines/`)

Then run this command in a shell:
  
```sh
R CMD javareconf
```

And these in R:

```R
install.packages('remotes')
library(remotes)
install_version('bartMachine', version='1.4.1.1')
install_version('dbarts', version='0.9-32')
install_github('rsparapa/bnptools', ref='1b3e608fc5a0345115e147cc18cd6e31d0b986b1', subdir='BART3')
```

Everything probably works with newer versions, but I've listed the ones I used to run the code myself for reproducibility.


## Python setup

Install `uv` (https://docs.astral.sh/uv/getting-started/installation/), check if it's already available on your system first. Then do

```sh
make setup
```


## How to run things

The commands are available through `uv run`, all commands have command line options that can be shown with `uv run <command> -h`. The results are saved in `./results` and the plots in `./plots`.

```sh
uv run benchmark       # clock a few iterations of bartz/dbarts/xgboost
uv run benchmark-plot  # plot the results of the above
uv run test-rmse       # compare the RMSE of BART packages (~2 hours)
uv run test-rmse-plot  # plot the results of the above
```

The plotting commands require results to be present for various combinations of options of the results-producing commands. For convenience, these `make` targets will run the commands for all configurations:

```sh
make benchmark-cpu  # repeat cpu benchmark for all configurations, ~1 hour

make benchmark-gpu  # repeat gpu benchmark for all configurations
make test-rmse      # repeat bart packages comparison for all configurations
```

Of these, only `make benchmark-gpu` requires a GPU.


## Troubleshooting

If there is a problem, [open a new issue on github](https://github.com/Gattocrucco/bart-gpu-article/issues).
