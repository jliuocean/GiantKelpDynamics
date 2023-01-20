include("shared.jl") # packages

using Oceananigans, StructArrays, Printf, JLD2, Statistics, CUDA, EasyFit
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_w_from_continuity!
using Oceananigans.Grids: xnodes, ynodes
using Oceananigans.Architectures: arch_array

using GiantKelpDynamics

function get_observable(path, observation_i, observation_j)
    u = FieldTimeSeries(path, "depth_average_u")

    reference_station_u = u[observation_i[1], observation_j[1], 1, :]

    reference_station_upstream = reference_station_u .>= 0
    reference_station_downstream = reference_station_u .< 0

    observables = []

    push!(observables, std(reference_station_u))

    middle_station_u = u[observation_i[2], observation_j[2], 1, :]

    push!(observables, std(middle_station_u))

    for station_idx in 3:length(observation_i)
        comparison_data_u = u[observation_i[I], observation_j[I], 1, :]

        std_u = std(comparison_data_u)

        push!(observables, std_u)

        comparison_data_east = comparison_data_u[reference_station_upstream]
        comparison_data_west = comparison_data_u[reference_station_downstream]

        upstream = fitlinear(reference_station_u[reference_station_upstream], comparison_data_east)
        downstream = fitlinear(reference_station_u[reference_station_downstream], comparison_data_west)

        push!(observables, upstream.a)

        push!(observables, downstream.a)
    end

    return observable
end

function run_member(id, generation, Cᵈᵇ, dropoff, Aᵤ)
    @info "$id, $generation, $Cᵈᵇ, $dropoff, $Aᵤ"
    filepath = "raw_results/calibration_ensemble_$(generation)_$(id)"

    arch = Oceananigans.CPU()
    FT = Float64

    # ## Setup grid 
    Lx, Ly, Lz = 3kilometers, 1kilometers, 8
    Nx, Ny, Nz = 3 * 256, 256, 8
    grid = RectilinearGrid(arch, FT;
                        size=(Nx, Ny, Nz), 
                        extent=(Lx, Ly, Lz))

    # ## Setup kelp particles

    forest_radius = 100

    xs = Vector{FT}()
    ys = Vector{FT}()
    sf = Vector{FT}()

    real_density = 0.5 # 1/m²
    grid_density = real_density * (Lx / Nx * Ly / Ny)
    node_density = 1
    base_scaling = node_density * grid_density

    for x in xnodes(Center, grid)[1:node_density:end], y in ynodes(Center, grid)[1:node_density:end]
        r = sqrt((x - Lx/2)^2 + (y - Ly/2)^2)
        if r < forest_radius
            scalefactor = base_scaling * (tanh((r + forest_radius * 0.9) / dropoff) - tanh((r - forest_radius * 0.9) / dropoff))/2
            push!(xs, x)
            push!(ys, y)
            push!(sf, scalefactor)
        end
    end

    number_nodes = 2
    segment_unstretched_length = [5.0, 4.0]

    kelps = GiantKelp(; grid, 
                        number_nodes, 
                        segment_unstretched_length, 
                        base_x = xs, base_y = ys, base_z = -8.0 * ones(length(xs)), 
                        initial_blade_areas = 3.0 .* [0.85, 0.15],
                        scalefactor = sf, 
                        architecture = arch, 
                        max_Δt = 0.4,
                        drag_fields = false,
                        parameters = (k = 10 ^ 5, 
                                    α = 1.41, 
                                    ρₒ = 1026.0, 
                                    ρₐ = 1.225, 
                                    g = 9.81, 
                                    Cᵈˢ = 1.0, 
                                    Cᵈᵇ= Cᵈᵇ, 
                                    Cᵃ = 3.0,
                                    n_nodes = number_nodes,
                                    τ = 1.0,
                                    kᵈ = 500))

    @inline tidal_forcing(x, y, z, t, params) = - params.Aᵤ * params.ω * sin(params.ω * (t - params.t_central) - params.ϕᵤ) - params.Aᵥ * params.ω * cos(params.ω * (t - params.t_central) - params.ϕᵥ)

    drag_set = DiscreteDragSet(; grid)

    Aᵥ = 0.0

    u_forcing = (Forcing(tidal_forcing, parameters = (Aᵤ = Aᵤ, Aᵥ = Aᵥ, ϕᵤ = -π/2, ϕᵥ = -π, t_central = 0, ω = 1.41e-4)), 
                drag_set.u)

    v_forcing = (Forcing(tidal_forcing, parameters = (Aᵤ = Aᵥ, Aᵥ = Aᵤ, ϕᵤ = -π, ϕᵥ = -π/2, t_central = 0, ω = 1.41e-4)),
                drag_set.v)

    w_forcing = drag_set.w

    u_bcs = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0.0))
    v_bcs = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0.0))
    w_bcs = FieldBoundaryConditions(bottom = OpenBoundaryCondition(0.0))

    model = NonhydrostaticModel(; grid,
                                timestepper = :RungeKutta3,
                                closure = AnisotropicMinimumDissipation(),
                                forcing = (u = u_forcing, v = v_forcing, w = w_forcing),
                                boundary_conditions = (u = u_bcs, v = v_bcs, w = w_bcs),
                                particles = kelps)

    Δt₀ = 0.5

    uᵢ(x, y, z) = Aᵤ * cos(π/2) + Aᵤ * (rand() - 0.5) * 2 * 0.01
    vᵢ(x, y, z) = Aᵥ * cos(π) * (1 + (rand() - 0.5) * 2 * 0.01)

    set!(model, u = uᵢ, v = vᵢ)

    # initialise kelp positions_ijk
    kelp_dynamics!(kelps, model, Δt₀)

    simulation = Simulation(model, Δt = Δt₀, stop_time = 1year)

    simulation.callbacks[:update_drag_fields] = Callback(drag_set; callsite = TendencyCallsite())

    wizard = TimeStepWizard(cfl = 0.8, max_change = 1.1, min_change = 0.8)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

    progress_message(sim) = @printf("Iteration: %07d, time: %s, Δt: %s, max(|u|) = %.1e ms⁻¹, min(|u|) = %.1e ms⁻¹, wall time: %s\n",
                                    iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                    maximum(abs, sim.model.velocities.u), minimum(abs, sim.model.velocities.u), prettytime(sim.run_wall_time))
        
    simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(1hour))

    depth_average_u = Average(model.velocities.u, dims = 3)
    depth_average_v = Average(model.velocities.v, dims = 3)

    simulation.output_writers[:profiles] =
        JLD2OutputWriter(model, (; depth_average_u, depth_average_v), filename = "$filepath.jld2", schedule = TimeInterval(5minute), overwrite_existing = true)

    simulation.stop_time = 2.5 * 2π / 1.41e-4

    run!(simulation)

    return simulation
end

function parameter_to_data_map(u, id, generation, observation_i, observation_j)
    final_state = run_member(id, generation, u["C"], u["dropoff"], u["A"])
    return get_observable(final_state.output_writers[:profiles].filepath, observation_i, observation_j)
end