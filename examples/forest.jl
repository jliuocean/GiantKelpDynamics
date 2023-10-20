# # [Single plant](@id single_example)
# In this example we setup a single plant in a narrow periodic channel to help understand the drag of the kelp on the water

# ## Install dependencies
# First we check we have the dependencies installed
# ```julia
# using Pkg
# pkg"add Oceananigans OceanBioME GiantKelpDynamics CairoMakie JLD2"
# ```

# Load the packages and setup the models

using Oceananigans, GiantKelpDynamics, OceanBioME, Oceananigans.Units
using OceanBioME: Biogeochemistry

grid = RectilinearGrid(size = (64, 32, 8), extent = (1kilometer, 500, 8))

xc, yc, zc = nodes(grid, Center(), Center(), Center())

x_spacing = xc[27]:xspacings(grid, Center()):xc[32]
y_spacing = yc[14]:yspacings(grid, Center()):yc[19]

holdfast_x = vec([x for x in x_spacing, y in y_spacing])
holdfast_y = vec([y for x in x_spacing, y in y_spacing])
holdfast_z = vec([-8. for x in x_spacing, y in y_spacing])

scalefactor = 0.5 * (xspacings(grid, Center()) * yspacings(grid, Center())) .* ones(length(holdfast_x))

scalefactor = vec([x for x in x_spacing, y in y_spacing])

number_nodes = 2

segment_unstretched_length = [16., 8.]

max_Δt = 1.

kelp = GiantKelp(; grid,
                   holdfast_x, holdfast_y, holdfast_z,
                   scalefactor, number_nodes, segment_unstretched_length,
                   max_Δt)

@inline sponge(x, y, z) = ifelse(x < 100, 1, 0)

u = Relaxation(; rate = 1/200, target = 0.2, mask = sponge)
v = Relaxation(; rate = 1/200, mask = sponge)
w = Relaxation(; rate = 1/200, mask = sponge)

model = NonhydrostaticModel(; grid, 
                              biogeochemistry = Biogeochemistry(NothingBGC(),
                                                                particles = kelp),
                              advection = WENO(),
                              forcing = (; u, v, w),
                              closure = AnisotropicMinimumDissipation())

# Set the initial positions of the plant nodes (relaxed floating to the surface), and the set an initial water velocity

set!(kelp, positions = [13.86 0. 8.; 21.86 0. 8.;])

set!(model, u = 0.05)

# Setup the simulaiton to save the flow and kelp positions

simulation = Simulation(model, Δt = 40, stop_time = 2hours)

prog(sim) = @info "Completed $(prettytime(time(simulation))) in $(simulation.model.clock.iteration) steps with Δt = $(prettytime(simulation.Δt))"

simulation.callbacks[:progress] = Callback(prog, IterationInterval(100))

simulation.output_writers[:flow] = JLD2OutputWriter(model, model.velocities, overwrite_existing = true, filename = "single_flow.jld2", schedule = TimeInterval(2minutes))
simulation.output_writers[:kelp] = JLD2OutputWriter(model, (; positions = kelp.positions), overwrite_existing = true, filename = "single_kelp.jld2", schedule = TimeInterval(2minutes))

# Run!
run!(simulation)

# Next we load the data
using CairoMakie, JLD2

u = FieldTimeSeries("single_flow.jld2", "u")

file = jldopen("single_kelp.jld2")

iterations = keys(file["timeseries/t"])

positions = [file["timeseries/positions/$it"] for it in iterations]

close(file)

times = u.times

# Now we can animate the motion of the plant and attenuation of the flow

n = Observable(1)

x_position_first = @lift vec([positions[$n][p][1, 1] for (p, x₀) in enumerate(holdfast_x)])
z_position_first = @lift vec([positions[$n][p][1, 3] for (p, z₀) in enumerate(holdfast_z)])

abs_x_position_first = @lift vec([positions[$n][p][1, 1] + x₀ for (p, x₀) in enumerate(holdfast_x)])
abs_z_position_first = @lift vec([positions[$n][p][1, 3] + z₀ for (p, z₀) in enumerate(holdfast_z)])

x_position_ends = @lift vec([positions[$n][p][2, 1] for (p, x₀) in enumerate(holdfast_x)])
y_position_ends = @lift vec([positions[$n][p][2, 2] for (p, y₀) in enumerate(holdfast_y)])

rel_x_position_ends = @lift vec([positions[$n][p][2, 1] - positions[$n][p][1, 1] for (p, x₀) in enumerate(holdfast_x)])
rel_z_position_ends = @lift vec([positions[$n][p][2, 3] - positions[$n][p][1, 3] for (p, z₀) in enumerate(holdfast_z)])

u_vert = @lift interior(u[$n], :, 16, :)

u_vert_lims = (minimum(u[:, 16, :, :]), maximum(u[:, 16, :, :]))

u_surface = @lift interior(u[$n], :, :, 8)

u_surface_lims = (minimum(u[:, :, 8, :]), maximum(u[:, :, 8, :]))

xf, yc, zc = nodes(u.grid, Face(), Center(), Center())

fig = Figure(resolution = (1200, 800));

ax = Axis(fig[1:3, 1], aspect = DataAspect())

hm = heatmap!(ax, xf, yc, u_surface, colorrange = u_surface_lims)

arrows!(ax, holdfast_x, holdfast_y, x_position_ends, y_position_ends, color = :black)

ax = Axis(fig[4, 1], limits = (400, 550, -8, 0), aspect = AxisAspect(20))

hm = heatmap!(ax, xf, zc, u_vert, colorrange = u_vert_lims)

arrows!(ax, holdfast_x, holdfast_z, x_position_first, z_position_first, color = :black)
arrows!(ax, abs_x_position_first, abs_z_position_first, rel_x_position_ends, rel_z_position_ends, color = :black)

record(fig, "drag.mp4", 1:length(times); framerate = 10) do i; 
    n[] = i
end

# ![](drag.mp4)
# In this video the limitations of the simplified drag stencil can be seen (see previous versions for a more complex stencil). It is better suited to the forest application like in the [forest example](@ref forest_example)
