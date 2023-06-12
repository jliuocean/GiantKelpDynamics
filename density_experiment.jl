using GiantKelpDynamics, CUDA, Oceananigans, Printf, JLD2
using Oceananigans.Units

using GiantKelpDynamics: segment_area_fraction

output_dir = joinpath(@__DIR__, ARGS[1])
member = parse(Int64, ARGS[2])

@load "final_densities.jld2" density

forest_density = density[member]

arch = CUDA.has_cuda_gpu() ? Oceananigans.GPU() : Oceananigans.CPU()
FT = CUDA.has_cuda_gpu() ? Float32 : Float64

segment_unstretched_length = [16, 8]

simulation = GiantKelpDynamics.setup_forest(; arch,
                                              Lx = 5kilometers, 
                                              Ly = 2kilometers, 
                                              forest_density, 
                                              number_nodes = 2, 
                                              segment_unstretched_length, 
                                              initial_pneumatocyst_volume = (2.5 / (5 * 9.81)) .* segment_area_fraction(segment_unstretched_length))

@info "setup forest"

wizard = TimeStepWizard(cfl = 0.7, max_change = 2.0, min_change = 0.5)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

@inline progress_message(sim) = @printf("Iteration: %07d, time: %s, Δt: %s\n",
                                        iteration(sim), prettytime(sim), prettytime(sim.Δt))
    

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(10))

simulation.output_writers[:profiles] =
    JLD2OutputWriter(simulation.model, merge(simulation.model.velocities, simulation.model.tracers),
                     filename = "$(output_dir)/$member.jld2",
                     schedule = TimeInterval(30minute),
                     overwrite_existing = true)

rm("$(member)_particles.jld2")

jldopen("$(member)_particles.jld2", "a+") do file
    file["base_position/x"] = simulation.model.particles.properties.x
    file["base_position/y"] = simulation.model.particles.properties.y
end          

@inline function store_particles!(simulation)
    jldopen("$(member)_particles.jld2", "a+") do file
        file["node_positions/$(simulation.model.clock.time)"] = simulation.model.particles.properties.positions
    end
end

simulation.callbacks[:particles] = Callback(store_particles!, TimeInterval(30minute))

simulation.stop_time = 2.5 * 2π / 1.41e-4

run!(simulation)