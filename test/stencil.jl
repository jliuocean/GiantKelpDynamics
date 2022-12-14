using Oceananigans, StructArrays
using Oceananigans.Units

using GiantKelpDynamics

kelp = GiantKelp(base_x = 4.0,
                 base_y = 4.0,
                 base_z = -8.0,
                 number_nodes = 3,
                 node_positions = [0.0 0.0 2.0; 1.0 0.0 4.0; 3.0 0.0 4.0],
                 segment_unstretched_length = 2.0,
                 node_stipe_radii = 0.03 * ones(3),
                 node_pneumatocyst_volumes = 0.002 * ones(3),
                 node_effective_radii = 0.5 * ones(3))

particle_struct = StructArray([kelp])

@inline guassian_smoothing(r, rᵉ) = exp(-(3 * r) ^ 2 / (2 * rᵉ^2)) / sqrt(2 * π * rᵉ ^ 2)

particles = LagrangianParticles(particle_struct; 
                            dynamics = kelp_dynamics!, 
                            parameters = (k = 10^5, 
                                          α = 1.41, 
                                          ρₒ = 1026.0, 
                                          ρₐ = 1.225, 
                                          g = 9.81, 
                                          Cᵈˢ = 1.0, 
                                          Cᵈᵇ= 0.4 * 12 ^ -0.485, 
                                          Cᵃ = 3.0,
                                          kᵈ = 10 ^ 4,
                                          drag_smoothing = guassian_smoothing,
                                          n_nodes = 3))

Lx, Ly, Lz = 8, 8, 8
Nx, Ny, Nz = 8 .*(Lx, Ly, Lz)
grid = RectilinearGrid(size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz), topology=(Periodic, Periodic, Bounded))

drag_nodes = CenterField(grid)

model = NonhydrostaticModel(;grid, particles, auxiliary_fields = (; drag_nodes))

set!(model, u=0.15)

drag_water_callback = Callback(drag_water!; callsite = TendencyCallsite())

simulation = Simulation(model, Δt=0.05, stop_time=1minutes)

simulation.callbacks[:drag_water] = drag_water_callback

#run!(simulation)