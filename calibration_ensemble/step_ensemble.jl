using LinearAlgebra, Random

using Distributions, Plots

using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses

rng_seed = 41
rng = Random.MersenneTwister(rng_seed)

dim_output = 4

σᵤ = [6.97 2.73]./100# 6.03 mean([3.9, 3.56, 2.69, 2.88]) mean([4.43, 3.81])] ./ 100
σᵥ = [1.11 0.53]./100# 1.21 mean([0.87, 0.65, 0.58, 0.66]) mean([0.92, 0.71])] ./ 100
gradient_east = [0.24]# 1.04 mean([0.44, 0.32, 0.24, 0.37]) mean([0.32, 0.35])]
r²_east = [0.43]# 0.76 mean([0.54, 0.48, 0.44, 0.46]) mean([0.38, 0.45])]
gradient_west = [0.51]# 0.57 mean([0.69, 0.67, 0.46, 0.39]) mean([0.105, 0.79])]
r²_west = [0.50]# 0.63 mean([0.47, 0.52, 0.51, 0.51]) mean([0.47, 0.43])]

n = 2016 # 15min intervals for 3 weeks
Γ_from_r(r, n) = 2 * r * (1 - r ^ 2) * (n - 1) / sqrt((n - 1) * (3 + n))

Γ_gradient_east = Γ_from_r.(sqrt.(r²_east), n) * 0.05
Γ_gradient_west = Γ_from_r.(sqrt.(r²_west), n) * 0.05
Γ_σᵤ = 0.05 .* σᵤ # complellty arbitaty

obs_data = [σᵤ..., gradient_east..., gradient_west...]
Γᵛ = [Γ_σᵤ..., Γ_gradient_east..., Γ_gradient_west...] .^ 2
Γ = zeros(dim_output, dim_output)
for i in 1:dim_output
    Γ[i, i] = Γᵛ[i]
end

noise_dist = MvNormal(zeros(dim_output), Γ)

y = obs_data .+ rand(noise_dist)

prior_Cᵈᵇ = constrained_gaussian("Cᵈᵇ", 0.4 * 12 ^ -0.485, 0.1 * 0.4 * 12 ^ -0.485, 0.01 * 0.4 * 12 ^ -0.485, Inf)
prior_dropoff = constrained_gaussian("dropoff", 10.0, 5.0, 1.0, 50.0)
prior_Aᵤ = constrained_gaussian("A", 0.2, 0.05, 0.05, 0.3)

prior = combine_distributions([prior_Cᵈᵇ, prior_dropoff, prior_Aᵤ])

generation = 1
generation_size = 6 # i.e. one days worth of running

results_file = jldopen("fawcett-results/ensemble_generation_$generation.jld2")
number_params = keys(results_file["parameters"])
generation_n_parameters = zeros(generation_size, sum(ones(Int, length(number_params))[number_params .!= "peak_density"]))

for (idx, parameter) in enumerate(number_params[number_params .!= "peak_density"]) # accidentally generted a peak desnity parameter but didn't read into model anyway
    generation_n_parameters[:, idx] = results_file["parameters/$parameter"][1:generation_size]
end

initial_ensemble = generation_n_parameters'

ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng)

G_ens = zeros(generation_size, dim_output)

# Solving the inverse problem