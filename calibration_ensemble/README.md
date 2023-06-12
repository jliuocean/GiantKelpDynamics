# Model calibration
Details of model calibration can be found in the accompanying publication. This path contains the files necessary to run the calibration, and instructions are below:

1. Clone this directory and instantiate the project:
```sh
> julia --project 
julia> using Pkg

julia> Pkg.instantiate()
```
2. Run `initialise_EKP.jl` with the arguments: 
    - output directory
    - data path (path to `truth.jld2`)
    - path to store the EKI
    - path to priors toml
    - number of ensemble members
    - random number generator seed
For example:
```
julia --project calibration_ensemble/initialize_EKP.jl results calibration_ensemble/truth.jld2 results calibration_ensemble/priors.toml 12 42
```
This will write various files.
4. Modify the slurm file `calibration_ensemble.sbatch` for your project location, threads, and `run_model.jl` arguments which should correspond to the output directory above and the ensemble generation (i.e. 1 here). For example:
```
julia --project --threads=14 calibraiton_ensemble/run_model.jl results 5 $SLURM_ARRAY_TASK_ID
```
And update the other slurm arguments.
5. Run the generation.
6. Run `calibration_ensemble/update_EKI.jl` with parameters of the same output directory, eki path and generation number as above. For example:
```
julia --project --threads=14 calibraiton_ensemble/update_EKI.jl results results 1
```
7. Update the slurm file for the next generation as in step 4 and run.
8. Repeat until parameters are sufficiently converged.


## `truth.jld2`

The truth values for calibration are derived from the observations of Gaylord, Brian, Rosman, Johanna H., Reed, Daniel C., Koseff, Jeffrey R., Fram, Jonathan, MacIntyre, Sally, Arkema, Katie, McDonald, Cameron, Brzezinski, Mark A., Largier, John L., Monismith, Stephen G., Raimondi, Peter T., Mardian, Brent, (2007), Spatial patterns of flow and their modification within and around a giant kelp forest, Limnology and Oceanography, 52, doi: 10.4319/lo.2007.52.5.1838. 