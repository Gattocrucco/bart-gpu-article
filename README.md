[![DOI](https://zenodo.org/badge/880171976.svg)](https://doi.org/10.5281/zenodo.14010504)

# bart-gpu-article

Code to reproduce the results in Petrillo (2024), "Very fast Bayesian Additive Regression Trees on GPU", [arXiv:2410.23244](https://arxiv.org/abs/2410.23244).

## Files setup

* Copy/clone the files to your computer
* Set the working directory to `bart-gpu-article/code`

## R setup

Install:

  * R 4.4.1 https://www.r-project.org
  
  * JDK 19.0.1 https://jdk.java.net (on macOS, put the directory into `/Library/Java/JavaVirtualMachines/`)

Then install the following R packages:

```R
library(remotes)
install_version('bartMachine', version='1.3.4.1')
install_version('BART', version='2.9.9')
install_version('dbarts', version='0.9-28')
```

Everything probably works with newer versions, but I've listed the ones I used to run the code myself for reproducibility.

## Python setup

### Brief version

Install Conda or equivalent and make the environment out of `condaenv.yml`.

### Long version

If you don't have Python experience, here is a possible detailed path to install everything without messing up other Python installations. On Mac; I don't know about Windows or Linux.

* Install [homebrew](https://brew.sh)
* `$ brew install micromamba`
* `$ micromamba env create -f condaenv.yml`
* `$ micromamba activate bart-gpu-article`

## How to run scripts

From the `code` directory, do

```sh
(bart-gpu-article) $ python script.py
```

Or, if you prefer to use IPython:

```sh:
(bart-gpu-article) $ pip install ipython
(bart-gpu-article) $ ipython
In [1]: run script.py
```

Each figure-producing script saves figures in a directory with the same name of the script.

## How to run notebooks

Google Colab allows to open notebooks from arbitrary github repositories. You'll have to fork out 10$ for the GPUs I use in the paper.

## Scripts & notebooks

* `speed-benchmark-*.py`: scripts to produce the CPU timings for figure 3.
* `speed-benchmark-*.ipynb`: notebooks to produce the GPU timings for figure 3.
* `speed-benchmark-plot.py`: makes figure 3; the results of the previous scripts have to be copy-pasted in this script.
* `test-rmse.py`: produce the data for figure 4.
* `test-rmse-plot.py`: make figure 4; the output of `test-rmse.py` has to be copy-pasted

## Troubleshooting

If there is a problem, [open a new issue on github](https://github.com/Gattocrucco/bart-gpu-article/issues).
