using Oceananigans, StructArrays, Printf, JLD2, Statistics, CUDA
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_w_from_continuity!
using Oceananigans.Grids: xnodes, ynodes
using Oceananigans.Architectures: arch_array

using GiantKelpDynamics

arch = CUDA.has_cuda_gpu() ? Oceananigans.GPU() : Oceananigans.CPU()
FT = CUDA.has_cuda_gpu() ? Float32 : Float64

if arch == Oceananigans.CPU()
    adapt_array(x) = x
else
    adapt_array(x) = CuArray(x)
end

@inline tidal_forcing(x, y, z, t, params) = - params.Aᵤ * params.ω * sin(params.ω * (t - params.t_central) - params.ϕᵤ) - params.Aᵥ * params.ω * cos(params.ω * (t - params.t_central) - params.ϕᵥ)

function setup_forest(arch ;
                      Lx = 3kilometers,
                      Ly = 1kilometers, 
                      Lz = 8,
                      horizontal_res = 256,
                      vertical_res = 1,
                      forest_radius = 100,
                      smoothing_distance = 16.6194891666667,
                      forest_density = 0.5, # 1/m²
                      node_density = 1, # one kelp per grid node
                      number_nodes = 2,
                      segment_unstretched_length = [7.8, 4.0],
                      parameters = (k = 1.91 * 10 ^ 7, 
                                    α = 1.41, 
                                    ρₒ = 1026.0, 
                                    ρₐ = 1.225, 
                                    g = 9.81, 
                                    Cᵈˢ = 1.0, 
                                    Cᵈᵇ= 0.87395175, 
                                    Cᵃ = 3.0,
                                    n_nodes = number_nodes,
                                    τ = 1.0,
                                    kᵈ = 500),
                      initial_blade_areas = 3.0 .* [0.8, 0.2],
                      initial_pneumatocyst_volume = (2.5 / (5 * 9.81)) .* ifelse(isa(segment_unstretched_length, Number), 1 / number_nodes .* ones(number_nodes), segment_unstretched_length ./ sum(segment_unstretched_length)),
                      Aᵤ = 0.103650,
                      Aᵥ = 0.0)

    Nx, Ny = Int.((Lx, Ly) .* horizontal_res ./ 1kilometer)
    Nz = Lz * vertical_res

    grid = RectilinearGrid(arch, FT;
                           size=(Nx, Ny, Nz), 
                           extent=(Lx, Ly, Lz), 
                           topology=(Periodic, Bounded, Bounded))

    xs = Vector{FT}()
    ys = Vector{FT}()
    sf = Vector{FT}()

    grid_density = forest_density * (Lx / Nx * Ly / Ny)
    base_scaling = node_density * grid_density

    for x in xnodes(Center, grid)[1:node_density:end], y in ynodes(Center, grid)[1:node_density:end]
        r = sqrt((x - Lx/2)^2 + (y - Ly/2)^2)
        if r < forest_radius
            scalefactor = base_scaling * (tanh((r + forest_radius * 0.9) / smoothing_distance) - tanh((r - forest_radius * 0.9) / smoothing_distance))/2
            push!(xs, x)
            push!(ys, y)
            push!(sf, scalefactor)
        end
    end

    kelps = GiantKelp(; grid, 
                        number_nodes, 
                        segment_unstretched_length, 
                        base_x = xs, base_y = ys, base_z = -8.0 * ones(length(xs)), 
                        initial_blade_areas,
                        scalefactor = sf, 
                        architecture = arch, 
                        max_Δt = 0.6,
                        drag_fields = false,
                        parameters,
                        initial_stretch = 1.0,
                        initial_pneumatocyst_volume)

    drag_set = DiscreteDragSet(; grid)
    tracer_exchange = TracerExchange(kelps, 10.0, 0.1)


    u_forcing = (Forcing(tidal_forcing, parameters = (Aᵤ = Aᵤ, Aᵥ = Aᵥ, ϕᵤ = -π/2, ϕᵥ = -π, t_central = 0, ω = 1.41e-4)), 
                drag_set.u)

    v_forcing = drag_set.v

    w_forcing = drag_set.w

    u_bcs = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0.0))
    v_bcs = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0.0))
    w_bcs = FieldBoundaryConditions(bottom = OpenBoundaryCondition(0.0))

    model = NonhydrostaticModel(; grid,
                                timestepper = :RungeKutta3,
                                advection = UpwindBiasedThirdOrder(),
                                closure = AnisotropicMinimumDissipation(),
                                forcing = (u = u_forcing, v = v_forcing, w = w_forcing),
                                boundary_conditions = (u = u_bcs, v = v_bcs, w = w_bcs),
                                particles = kelps, 
                                tracers = (:U, :O)) #takeUp, Outputted

    uᵢ(x, y, z) = Aᵤ * cos(π/2) + Aᵤ * (rand() - 0.5) * 2 * 0.01

    set!(model, u = uᵢ, U = 10)

    Δt₀ = 0.5
    # initialise kelp positions_ijk
    kelp_dynamics!(kelps, model, Δt₀)

    simulation = Simulation(model, Δt = Δt₀, stop_time = 1year)

    simulation.callbacks[:update_drag_fields] = Callback(drag_set; callsite = TendencyCallsite())
    simulation.callbacks[:tracer_exchange] = Callback(tracer_exchange; callsite = TendencyCallsite())

    return simulation
end