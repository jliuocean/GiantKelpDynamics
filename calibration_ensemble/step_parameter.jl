using Tensorial, JLD2, Statistics, Distributions

@inline function randn(μ, σ, lb, ub, sz)
    d = Truncated(Normal(μ, σ), lb, ub)
    return rand(d, sz)
end

function step_parameter(parameters, observables, observed_data)
    parameter_means = mean(parameters, dims=1)[1, :]
    observable_means = mean(observables, dims=1)[1, :]

    Δobs = zeros(size(observables))
    Δparams = zeros(size(parameters))

    for i in 1:size(parameters)[1]
        Δobs[i, :] = observables[i, :] - observable_means
        Δparams[i, :] = parameters[i, :] - parameter_means
    end

    Δobs = Tensor{Tuple{size(observables)...}, Float64}(Δobs)
    Δparams = Tensor{Tuple{size(parameters)...}, Float64}(Δparams)

    cov_pp = zeros(size(observables)[2], size(observables)[2])
    cov_up = zeros(size(observables)[2], size(parameters)[2])

    for i in 1:size(parameters)[1]
        cov_pp += Δobs[i, :] ⊗ Δobs[i, :]
        cov_up += Δparams[i, :] ⊗ Δobs[i, :]
    end

    next_generation_data = observed_data + randn(0, Γ / h, -Inf, Inf, size(observed_data))

    new_parameters = similar(parameters)
    for i in 1:size(observables)[1]
        r = next_generation_data - observables[i, :]
        new_parameters[i, :] = parameters[i, :] + cov_up' \ (cov_pp .+ Γ / h) * r
    end

    return new_parameters
end
const Γ, h = 1, 1/10

const obs_data = [6.97 2.73 6.03 mean([3.9, 3.56, 2.69, 2.88]) mean([4.43, 3.81]);
                  1.11 0.53 1.21 mean([0.87, 0.65, 0.58, 0.66]) mean([0.92, 0.71]);
                  1 0.24 1.04 mean([0.44, 0.32, 0.24, 0.37]) mean([0.32, 0.35]);
                  1 0.43 0.76 mean([0.54, 0.48, 0.44, 0.46]) mean([0.38, 0.45]);
                  1 0.51 0.57 mean([0.69, 0.67, 0.46, 0.39]) mean([0.105, 0.79]);
                  1 0.50 0.63 mean([0.47, 0.52, 0.51, 0.51]) mean([0.47, 0.43])]

generation = 1
generation_size = 12 # i.e. one days worth of running

results_file = jldopen("ensemble_generation_$generation.jld2")
number_params = keys(results_file["parameters"])
generation_n_parameters = zeros(generation_size, length(number_params))

for (idx, parameter) in enumerate(number_params)
    generation_n_parameters[:, idx] = results_file["parameters/$parameter"]
end

# probably need to come up with some way to handel a member not finish which would break this and presumably change the results?

generation_n_observables = zeros(generation_size, size(obs_data)...)
for id in 1:generation_size
    generation_n_observables[id, :, :] = results_file["$id"]
end

generation_np1_parameters = step_parameter(parameters, observables, observed_data)

jldopen("ensemble_generation_$(generation + 1).jld2", "w+") do file
    for (idx, parameter) in enumerate(number_params)
        file["parameters/$parameter"] = generation_np1_parameters[:, idx]
    end
end
