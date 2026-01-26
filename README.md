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

Issue `make` without arguments in a shell to get a list of the available commands.


## Troubleshooting

If there is a problem, [open a new issue on github](https://github.com/Gattocrucco/bart-gpu-article/issues).
