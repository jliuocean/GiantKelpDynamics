using GiantKelpDynamics, CUDA, Oceananigans, Printf, JLD2
using Oceananigans.Units

using GiantKelpDynamics: segment_area_fraction

arch = CUDA.has_cuda_gpu() ? Oceananigans.GPU() : Oceananigans.CPU()
FT = CUDA.has_cuda_gpu() ? Float32 : Float64

segment_unstretched_length = [16, 8]

simulation = GiantKelpDynamics.setup_forest(; arch,
                                              Lx = 3kilometers, 
                                              Ly = 1kilometers, 
                                              forest_density = 0.1, 
                                              number_nodes = 2, 
                                              segment_unstretched_length, 
                                              initial_pneumatocyst_volume = (2.5 / (5 * 9.81)) .* segment_area_fraction(segment_unstretched_length))

@info "setup forest"

wizard = TimeStepWizard(cfl = 0.8, max_change = 1.1, min_change = 0.8)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

progress_message(sim) = @printf("Iteration: %07d, time: %s, Δt: %s, max(|u|) = %1.1e ms⁻¹, min(|u|) = %.1e ms⁻¹, wall time: %s, min(|U|) = %.1e , max(|O|) = %.1e \n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                maximum(abs, sim.model.velocities.u), minimum(abs, sim.model.velocities.u), prettytime(sim.run_wall_time),
                                minimum(sim.model.tracers.U), maximum(sim.model.tracers.O))
    

simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(5minute))

simulation.output_writers[:profiles] =
    JLD2OutputWriter(simulation.model, merge(simulation.model.velocities, simulation.model.tracers),
                     filename = "fix_subduction.jld2",
                     schedule = TimeInterval(5minute),
                     overwrite_existing = true)

function store_particles!(sim)
    jldopen("fix_subduction_particles.jld2", "a+") do file
        file["particles/$(sim.model.clock.time)"] = sim.model.particles.properties
    end
end

simulation.callbacks[:save_particles] = Callback(store_particles!, TimeInterval(5minute))

simulation.stop_time = 3.095hours

subducted = [0]

function get_min!(simulation, subducted)
    x⃗ = simulation.model.particles.properties.positions

    subducted[1] += sum([x⃗[j][1, 3] for j in 1:length(simulation.model.particles)] .< 7.5)
end

simulation.callbacks[:min_length] = Callback(get_min!, IterationInterval(10); parameters = subducted)

run!(simulation)