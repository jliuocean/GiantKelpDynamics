using GiantKelpDynamics, CUDA, Oceananigans, Printf, JLD2
using Oceananigans.Units

using GiantKelpDynamics: segment_area_fraction

output_dir = joinpath(@__DIR__, ARGS[1])
member = parse(Int64, ARGS[2])

@load "density_experiment.jld2" density

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

wizard = TimeStepWizard(cfl = 0.8, max_change = 1.1, min_change = 0.8)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

progress_message(sim) = @printf("Iteration: %07d, time: %s, Δt: %s, max(|u|) = %1.1e ms⁻¹, min(|u|) = %.1e ms⁻¹, wall time: %s, max(|O|) = %.1e \n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                maximum(abs, sim.model.velocities.u), minimum(abs, sim.model.velocities.u), prettytime(sim.run_wall_time),
                                maximum(sim.model.tracers.O))
    

simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(5minute))

simulation.output_writers[:profiles] =
    JLD2OutputWriter(simulation.model, merge(simulation.model.velocities, simulation.model.tracers),
                     filename = "fix_subduction.jld2",
                     schedule = TimeInterval(30minute),
                     overwrite_existing = true)

simulation.stop_time = 2.5 * 2π / 1.41e-4

run!(simulation)
