using Oceananigans, StructArrays, Printf, JLD2, Statistics, CUDA, EasyFit
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_w_from_continuity!
using Oceananigans.Grids: xnodes, ynodes
using Oceananigans.Architectures: arch_array

using GiantKelpDynamics

function get_observable(simulation, observed_locations)
    path = simulation.output_writers[:profiles].filepath

    u = FieldTimeSeries(path, "depth_average_u")
    v = FieldTimeSeries(path, "depth_average_v")

    reference_station_u = u[observed_locations[1, 1], observed_locations[2, 1]]

    reference_station_upstream = reference_station_u .>= 0
    reference_station_downstream = reference_station_u .< 0

    observable = zeros(3, length(observed_locations[1, :]))

    for station_idx in 1:length(observed_locations[1, :])
        comparison_data_u = u[observed_locations[1, idx], observed_locations[2, idx]]
        #comparison_data_v = v[observed_locations[1, idx], observed_locations[2, idx]]

        std_u = std(comparison_data_u)
        #std_v = std(comparison_data_v)

        if station_idx != 1
            comparison_data_east = comparison_data_u[reference_station_upstream]
            comparison_data_west = comparison_data_u[reference_station_downstream]

            upstream = fitlinear(reference_station_u[reference_station_upstream], comparison_data_east)
            downstream = fitlinear(reference_station_u[reference_station_downstream], comparison_data_west)

            upstream_gradient = upstream.about
            #upstream_r² = upstream.R^2

            downstream_gradient = downstream.about
            #downstream_r² = downstream.R^2
        else
            upstream_gradient, upstream_r², downstream_gradient, downstream_r² = 1.0, 1.0, 1.0, 1.0
        end

        observable[:, idx] = [std_u, upstream_gradient, downstream_gradient]
    end

    return observable
end

generation, id = parse.(Int, ARGS)

file = jldopen("ensemble_generation_$generation.jld2")
Cᵈᵇ, peak_density, dropoff, Aᵤ = @show [file["parameters/$symbol"][id] for symbol in (:Cᵈᵇ, :peak_density, :dropoff, :Aᵤ)]
close(file)

#Cᵈᵇ, peak_density, dropoff, Aᵤ = parse.(Float64, (id, Cᵈᵇ, peak_density, dropoff, Aᵤ, generation))

filepath = "calibration_ensemble_$(generation)_$(id)"

arch = Oceananigans.CPU()
FT = Float64

# ## Setup grid 
Lx, Ly, Lz = 1kilometers, 1kilometers, 8
Nx, Ny, Nz = 256, 256, 8
grid = RectilinearGrid(arch, FT;
                       size=(Nx, Ny, Nz), 
                       extent=(Lx, Ly, Lz),
                       topology=(Periodic, Bounded, Bounded))

# ## Setup kelp particles

forest_radius = 100

xs = Vector{FT}()
ys = Vector{FT}()
sf = Vector{FT}()

real_density = 0.5 # 1/m²
grid_density = real_density * (Lx / Nx * Ly / Ny)
node_density = 1
base_scaling = node_density * grid_density * peak_density

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

# I think this ω gives a period of 1 day but it should be 12 hours?
u_forcing = (Forcing(tidal_forcing, parameters = (Aᵤ = Aᵤ, Aᵥ = 0.05, ϕᵤ = -π/2, ϕᵥ = -π, t_central = 0, ω = 1.41e-4)), 
             drag_set.u)

v_forcing = (Forcing(tidal_forcing, parameters = (Aᵤ = 0.05, Aᵥ = Aᵤ, ϕᵤ = -π, ϕᵥ = -π/2, t_central = 0, ω = 1.41e-4)),
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
vᵢ(x, y, z) = 0.05 * cos(π) * (1 + (rand() - 0.5) * 2 * 0.01)

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

depth_average_u = mean(model.velocities.u, dims = 3)
depth_average_v = mean(model.velocities.v, dims = 3)

simulation.output_writers[:profiles] =
    JLD2OutputWriter(model, (; depth_average_u, depth_average_v), filename = "$filepath.jld2", schedule = TimeInterval(5minute), overwrite_existing = true)

simulation.stop_time = 2.5 * 2π / 1.41e-4

run!(simulation)

obs_locs = [51 115 115 128 141;
            102 128 115 128 128]
            
observables = get_observable(simulation, obs_locs)

jldopen("ensemble_generation_$generation.jld2", "a+") do file
    file["$id"] = observables
end
