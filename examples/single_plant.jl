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

grid = RectilinearGrid(size = (256, 32, 32), extent = (100, 8, 8))

holdfast_x = [20.]
holdfast_y = [4.]
holdfast_z = [-8.]

kelp = GiantKelp(; grid,
                    holdfast_x, holdfast_y, holdfast_z)

model = NonhydrostaticModel(; grid, 
                              biogeochemistry = Biogeochemistry(NothingBGC(),
                                                                particles = kelp),
                              advection = WENO(),
                              closure = AnisotropicMinimumDissipation())

# Set the initial positions of the plant nodes (relaxed floating to the surface), and the set an initial water velocity

set!(kelp, positions = [0. 0. 3.; 0. 0. 6.; 0. 0. 8.; 3. 0. 8.; 6. 0. 8.; 9. 0. 8.; 12. 0. 8.; 15. 0. 8.;])

set!(model, u = 0.2)

# Setup the simulaiton to save the flow and kelp positions

simulation = Simulation(model, Δt = 0.5, stop_time = 10minutes)

prog(sim) = @info "Completed $(prettytime(time(simulation))) in $(simulation.model.clock.iteration) steps with Δt = $(prettytime(simulation.Δt))"

simulation.callbacks[:progress] = Callback(prog, IterationInterval(100))

simulation.output_writers[:flow] = JLD2OutputWriter(model, model.velocities, overwrite_existing = true, filename = "single_flow.jld2", schedule = TimeInterval(10))
simulation.output_writers[:kelp] = JLD2OutputWriter(model, (; positions = kelp.positions), overwrite_existing = true, filename = "single_kelp.jld2", schedule = TimeInterval(10))

# Run!

run!(simulation)

# Next we load the data
using CairoMakie, JLD2

u = FieldTimeSeries("single_flow.jld2", "u")

file = jldopen("single_kelp.jld2")

iterations = keys(file["timeseries/t"])

positions = [file["timeseries/positions/$it"][1] for it in iterations]

close(file)

times = u.times

# Now we can animate the motion of the plant and attenuation of the flow

n = Observable(1)

x_position = @lift positions[$n][:, 1] .+ 20
y_position = @lift positions[$n][:, 2] .+ 4
z_position = @lift positions[$n][:, 3] .- 8

u_vert = @lift interior(u[$n], :, 16, :)

u_vert_lims = (minimum(u[:, 16, :, :]), maximum(u[:, 16, :, :]))

u_surface = @lift interior(u[$n], :, :, 8)

u_surface_lims = (minimum(u[:, :, 8, :]), maximum(u[:, :, 8, :]))

xf, yc, zc = nodes(u.grid, Face(), Center(), Center())

fig = Figure(resolution = (1200, 400));

ax = Axis(fig[1, 1], aspect = DataAspect())

hm = heatmap!(ax, xf, zc, u_vert, colorrange = u_vert_lims)

scatter!(ax, x_position, z_position)

ax = Axis(fig[2, 1], aspect = DataAspect())

hm = heatmap!(ax, xf, yc, u_surface, colorrange = u_surface_lims)

scatter!(ax, x_position, y_position)

record(fig, "drag.mp4", 1:length(times); framerate = 10) do i; 
    n[] = i
end

# ![](drag.mp4)
# In this video the limitations of the simplified drag stencil can be seen (see previous versions for a more complex stencil). It is better suited to the forest application like in the [forest example](@ref forest_example)
