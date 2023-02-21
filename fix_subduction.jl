include("kelp_forest.jl")

simulation = setup_forest(arch; Lx = 3kilometers, Ly = 1kilometers, forest_density = 0.1, number_nodes = 3, segment_unstretched_length = [8, 4, 4], initial_blade_areas = 3.0 .* [.5, .3, .2], initial_pneumatocyst_volume = (2.5 / (5 * 9.81)) .* [0.0, 0.5, 0.5])

@info "setup forest"

wizard = TimeStepWizard(cfl = 0.8, max_change = 1.1, min_change = 0.8)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

progress_message(sim) = @printf("Iteration: %07d, time: %s, Δt: %s, max(|u|) = %.1e ms⁻¹, min(|u|) = %.1e ms⁻¹, wall time: %s, min(|U|) = %.1e , max(|O|) = %.1e \n",
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

simulation.stop_time = 1.0 * 2π / 1.41e-4

run!(simulation)