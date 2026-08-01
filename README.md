[![DOI](https://zenodo.org/badge/880171976.svg)](https://doi.org/10.5281/zenodo.14010504)

# bart-gpu-article

Code to reproduce the results in Petrillo (2024), "Very fast Bayesian Additive Regression Trees on GPU", [arXiv:2410.23244](https://arxiv.org/abs/2410.23244). To run anything, first clone this repository to your computer, and make it the current working directory in a terminal.


## Partial reproduction (only regenerate the plots)

The results used to generate the plots are saved in this repository so the plots can be re-generated without producing the results.


### Setup

Make sure `uv` (https://docs.astral.sh/uv/getting-started/installation/) is installed. It's a Python manager.


### Generate the plots as they appear in the article

```sh
make article-plots
make copy-plots  # this copies the plots to the article/ dir for the tex
```


### Generate arbitrary plots from the results

Check out the help of these commands:

```sh
uv run benchmark-plot --help
uv run compare-bart-packages-plot --help
uv run fullbench-plot --help
```


## Full reproduction

### Hardware setup

Running the GPU benchmarks requires a NVIDIA GPU. Virtual machines with GPUs can be conveniently rented in places like https://cloud.vast.ai starting from about 0.10 $/hour. The GPU used in the article is an RTX PRO 5000 which currently costs about 0.70 $/hour and has 48 GiB RAM.

Other things can be run on any computer with at least 16 GiB of RAM.


### R setup

R is needed only to run benchmarks involving R packages, and in particular java is needed only for `bartMachine`, which is used only in the `compare-bart-packages` command.

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


### Python setup

Install `uv` (https://docs.astral.sh/uv/getting-started/installation/), check if it's already available on your system first. Then do

```sh
make setup
```


### Iteration timing

```sh
make benchmark-cpu  # ~2 hours
```

Then, on a machine with an nvidia gpu:

```sh
make benchmark-gpu  # ~2 hours
```

Copy all the result files to the same machine.


### Comparison of BART packages

GPU not needed for this one.

```sh
make compare-bart-packages  # ~2 hours
```


### End-to-end benchmark on datasets

This requires a machine with at least ~64 GiB cpu ram, ~32 GiB gpu ram, and ~64 GiB disk.

```sh
make fullbench-data  # downloads data from OpenML
make fullbench-gpu  # ~6 hours
```

This one does not have a cpu variant, it would be too slow.


## Troubleshooting

If there is a problem, [open a new issue on github](https://github.com/Gattocrucco/bart-gpu-article/issues).
