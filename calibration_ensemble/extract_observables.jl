include("ensemble_member.jl")

using TOML, JLD2

function path_ensemble(load_path, iteration, member; pad_zeros = 3)

    load_dir = joinpath(load_path, join(["iteration", lpad(iteration, pad_zeros, "0")], "_"))
    subdir_name = join(["member", lpad(member, pad_zeros, "0")], "_")

    return joinpath(load_dir, subdir_name)

end

get_param_values(param_dict::Dict, names) = Dict(n => param_dict[n]["value"] for n in names)

function main()
    # Paths
    output_dir = joinpath(@__DIR__, ARGS[1])
    iteration = parse(Int64, ARGS[2])
    member = parse(Int64, ARGS[3])

    # equiv to 1, 2, 4, 8 (2 to catch the shaddow slowdown correctly)
    # We will observe σᵤ at all, and the gradient difference between 1 vs 4 and 1 vs 8
    observation_i = [76, 76, 110, 128] .+ 256 * 2
    observation_j = [75, 128, 128, 128]

    # get parameters
    member_path = path_ensemble(output_dir, iteration, member)
    param_dict = TOML.parsefile(joinpath(member_path, "parameters.toml"))
    names = ["C", "dropoff", "A"]
    params = @show get_param_values(param_dict, names)

    # evaluate map with noise to create data
    model_output = get_observable("raw_results/calibration_ensemble_$(iteration)_$(member).jld2", observation_i, observation_j)

    output_path = joinpath(member_path, "output.jld2")
    @save output_path model_output
end

main()