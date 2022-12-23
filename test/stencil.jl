using Oceananigans, StructArrays
using Oceananigans.Units

using GiantKelpDynamics

Lx, Ly, Lz = 8, 8, 8
Nx, Ny, Nz = 8 .*(Lx, Ly, Lz)
grid = RectilinearGrid(size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz), topology=(Periodic, Periodic, Bounded))

@inline guassian_smoothing(r, rᵉ) = exp(-(3 * r) ^ 2 / (2 * rᵉ^2)) / sqrt(2 * π * rᵉ ^ 2)

kelp = GiantKelp(; grid, 
                   base_x = [4.0], 
                   base_y = [4.0], 
                   base_z = [-8.0],
                   number_nodes = 3,
                   initial_node_positions = [0.0 0.0 2.0; 1.0 0.0 4.0; 3.0 0.0 4.0],
                   segment_unstretched_length = 2.0,
                   initial_stipe_radii = 0.03,
                   initial_pneumatocyst_volume = 0.002 * ones(3),
                   initial_effective_radii = 0.5 * ones(3),
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
                                 n_nodes = 3,
                                 τ = 5.0))

model = NonhydrostaticModel(;grid, particles = kelp)

set!(model, u=0.15)

drag_water_callback = Callback(fully_resolved_drag!; callsite = TendencyCallsite())

simulation = Simulation(model, Δt=0.05, stop_iteration=1)

simulation.callbacks[:drag_water] = drag_water_callback

#run!(simulation)

# # To Jago of the future, to check the stencil, run!, then plot:
# julia ```
# fig = Figure(resolution = (1600, 1000))
# ax = Axis(fig[1, 1])
# hm = heatmap!(ax, xnodes(Center, grid)[1:Nx], znodes(Center, grid)[1:Nz], kelp.properties.drag_field[1][1:Nx, 32, 1:Nz])
# scatter!(ax, model.particles.properties.positions[1][:, 1] .+ model.particles.properties.x[1], model.particles.properties.positions[1][:, 3] .+ model.particles.properties.z[1])
# ```
# This should give a horizontal line around the last node, then change `for n = 1:n_nodes` to `for n = 1:n_nodes -1`
# And replot which should give an angled line around the second node, then repeat for `- 2` which should give a vertical line from the bottom to the mid point of the first and second nodes