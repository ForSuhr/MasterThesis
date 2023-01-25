# Introduction

This is the repository of the master thesis "On the identification of port-Hamiltonian models via machine-learning".

# Get Started

All scripts are sorted and grouped according to the chapters and sections numbering in the thesis. A part of the code is also included in the thesis, but the code in this repository contains more details.

## Environment Setup

1. Install Julia and Julia plugin in vscode marketplace. Add Julia to path.
2. Open Julia REPL (with vscode terminal) and change directory to this repo. Press `]` to get into package mode. Run `activate .`
3. Run `instantiate` to install dependencies.

Please note that `instantiate` is required only once, while `activate .` is required every time you run this project. To activate the project automatically, create a startup file `startup.jl` under the folder `~/.julia/config/startup.jl` where you installed Julia. Paste the following code in the startup file:

```
using Pkg

if isfile("Project.toml") && isfile("Manifest.toml")
    Pkg.activate(".")
end
```

## Usage

### Run Code

To run a code block (or code line), simply click on the code block and press `shift + enter`. Since Julia is JIT (just-in-time) compiled, only this code block will be compiled and executed. The variables can be found in the workspace. They will not be deleted unless you close the Julia REPL.

### Scripts

It is recommended to start with [7.3_Experiment_idho](https://git.ltd.uni-erlangen.de/JiandongZhao/master-thesis/-/blob/main/src/examples_in_thesis/Chapter%207%20Compositional%20Modelling/7.3_Experiment_idho.jl). In this script, we illustrate the training and evaluation step by step.

Under the folder [Chapter_7_Compositional_Modelling](https://git.ltd.uni-erlangen.de/JiandongZhao/master-thesis/-/tree/main/src/examples_in_thesis/Chapter%207%20Compositional%20Modelling), there are four folders:

- figures: stores the result plots.
- helpers: provides some useful helper functions for reuse.
- models: stores the trained neural network models.
- parameters: stores the parameters of the neural networks.

# Links

- [Compiled pdf file of the thesis](https://git.ltd.uni-erlangen.de/JiandongZhao/master-thesis/-/blob/main/master_thesis/compiled_files)
- [Code in the thesis](https://git.ltd.uni-erlangen.de/JiandongZhao/master-thesis/-/tree/main/src)
