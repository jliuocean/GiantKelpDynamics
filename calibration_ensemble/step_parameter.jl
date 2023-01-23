using Tensorial, JLD2, Statistics, Distributions

@inline function randn(μ, σ, lb, ub, sz)
    d = Truncated(Normal(μ, σ), lb, ub)
    return rand(d, sz)
end

@inline function randn(μ, σ, lb, ub)
    d = Truncated(Normal(μ, σ), lb, ub)
    return rand(d)
end

function step_parameter(parameters, observables, observed_data)
    n_members = size(parameters)[1]

    parameter_means = mean(parameters, dims=1)[1, :]
    observable_means = mean(observables, dims=1)[1, :]

    Δobs = zeros(size(observables))
    Δparams = zeros(size(parameters))

    for i in 1:n_members
        Δobs[i, :] = observables[i, :] - observable_means
        Δparams[i, :] = parameters[i, :] - parameter_means
    end

    Δobs = Tensor{Tuple{size(observables)...}, Float64}(Δobs)
    Δparams = Tensor{Tuple{size(parameters)...}, Float64}(Δparams)

    cov_pp = zeros(size(observables)[2], size(observables)[2])
    cov_up = zeros(size(observables)[2], size(parameters)[2])

    for i in 1:n_members
        cov_pp += (Δobs[i, :] ⊗ Δobs[i, :]) ./ n_members
        cov_up += (Δparams[i, :] ⊗ Δobs[i, :]) ./ n_members
    end

    next_generation_data = observed_data .+ randn.(0, Γ ./ h, -Inf, Inf)

    @info size(cov_up)

    new_parameters = similar(parameters)
    for i in 1:n_members
        r = @show next_generation_data - observables[i, :]
        for p in 1:size(parameters)[2]
            new_parameters[i, p] = parameters[i, p] + (cov_up' \ (cov_pp .+ Γ[p] / h) * r)[p]
        end
    end

    return new_parameters
end

h = 1/10
# 1, 4, 3, mean(6, 7, 8, 9), mean(11, 12)
σᵤ = [6.97 2.73]./100# 6.03 mean([3.9, 3.56, 2.69, 2.88]) mean([4.43, 3.81])] ./ 100
σᵥ = [1.11 0.53]./100# 1.21 mean([0.87, 0.65, 0.58, 0.66]) mean([0.92, 0.71])] ./ 100
gradient_east = [1 0.24]# 1.04 mean([0.44, 0.32, 0.24, 0.37]) mean([0.32, 0.35])]
r²_east = [1 0.43]# 0.76 mean([0.54, 0.48, 0.44, 0.46]) mean([0.38, 0.45])]
gradient_west = [1 0.51]# 0.57 mean([0.69, 0.67, 0.46, 0.39]) mean([0.105, 0.79])]
r²_west = [1 0.50]# 0.63 mean([0.47, 0.52, 0.51, 0.51]) mean([0.47, 0.43])]

n = 2016 # 15min intervals for 3 weeks

Γ_from_r(r, n) = 2 * r * (1 - r ^ 2) * (n - 1) / sqrt((n - 1) * (3 + n))

Γ_gradient_east = Γ_from_r.(sqrt.(r²_east), n)
Γ_gradient_west = Γ_from_r.(sqrt.(r²_west), n)
Γ_σᵤ = 0.01 .* σᵤ # complellty arbitaty

obs_data = [σᵤ..., gradient_east..., gradient_west...]
Γ = [Γ_gradient_east..., Γ_gradient_west..., Γ_σᵤ...]

generation = 1
generation_size = 12 # i.e. one days worth of running

results_file = jldopen("fawcett-results/ensemble_generation_$generation.jld2")
number_params = keys(results_file["parameters"])
generation_n_parameters = zeros(generation_size, sum(ones(Int, length(number_params))[number_params .!= "peak_density"]))

for (idx, parameter) in enumerate(number_params[number_params .!= "peak_density"]) # accidentally generted a peak desnity parameter but didn't read into model anyway
    generation_n_parameters[:, idx] = results_file["parameters/$parameter"][1:generation_size]
end

# probably need to come up with some way to handel a member not finish which would break this and presumably change the results?
# because failing to finish is its self a result (that the conditions are too extreme and unphysical?)

generation_n_observables = zeros(generation_size, size(obs_data)...)
for id in 1:generation_size
    generation_n_observables[id, :] = results_file["$id"][:, 1:2]'
end

generation_np1_parameters = step_parameter(generation_n_parameters, generation_n_observables, obs_data)

#=
jldopen("ensemble_generation_$(generation + 1).jld2", "w+") do file
    for (idx, parameter) in enumerate(number_params)
        file["parameters/$parameter"] = generation_np1_parameters[:, idx]
    end
end
=#