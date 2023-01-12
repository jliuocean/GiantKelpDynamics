using Distributions, JLD2
import Base: randn

function write_member_start_file(id, Cᵈᵇ, peak_density, dropoff, Aᵤ, generation)
    file_text = "#!/bin/bash
#SBATCH -p skylake
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00

julia --project=/nfs/st01/hpc-atmos-jrt51/js2430/KelpPhysics --threads=12 ensemble_member.jl $id $Cᵈᵇ $peak_density $dropoff $Aᵤ $generation"
    open("generation_$(generation)_id_$id.sh", "w+") do file
        write(file, file_text)
    end
end

@inline function randn(μ, σ, lb, ub, sz)
    d = Truncated(Normal(μ, σ), lb, ub)
    return rand(d, sz)
end

ensemble_size = 100

generation = 1

Cᵈᵇ = randn(0.4 * 12 ^ -0.485, 0.1 * 0.4 * 12 ^ -0.485, 0.01 * 0.4 * 12 ^ -0.485, Inf, (ensemble_size, ))
peak_density = randn(1, 0.5, 0.1, Inf, (ensemble_size, ))
dropoff = randn(10.0, 5.0, 1.0, 50.0, (ensemble_size, ))
Aᵤ = randn(0.2, 0.05, 0.05, 0.5, (ensemble_size, ))

jldopen("ensemble_generation_$generation.jld2", "w+") do file
    file["parameters/Cᵈᵇ"] = Cᵈᵇ
    file["parameters/peak_density"] = peak_density
    file["parameters/dropoff"] = dropoff
    file["parameters/Aᵤ"] = Aᵤ
end
runfile = "#!/bin/bash"

for id in 1:ensemble_size
    write_member_start_file(id, Cᵈᵇ[id], peak_density[id], dropoff[id], Aᵤ[id], generation)
    global runfile *= "\nsbatch generation_$(generation)_id_$id.sh"
end

open("run_$generation.sh", "w+") do file
    write(file, runfile)
end
