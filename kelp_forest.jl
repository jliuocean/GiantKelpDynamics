using Oceananigans, StructArrays, Printf, JLD2, Statistics, CUDA
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_w_from_continuity!
using Oceananigans.Grids: xnodes, ynodes
using Oceananigans.Architectures: arch_array

using GiantKelpDynamics

arch = CUDA.has_cuda_gpu() ? Oceananigans.GPU() : Oceananigans.CPU()

if arch == Oceananigans.CPU()
    adapt_array(x) = x
else
    adapt_array(x) = CuArray(x)
end
# ## Setup grid 
Lx, Ly, Lz = 1kilometers, 1kilometers, 8
Nx, Ny, Nz = 256, 256, 8
grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz), topology=(Periodic, Periodic, Bounded))

# ## Setup kelp particles

forest_radius = 100
smoothing_disance = 3.0
#=
kelps = GiantKelp[]

for x in xnodes(Center, grid), y in ynodes(Center, grid)
    r = sqrt((x - Lx/2)^2 + (y - Ly/2)^2)
    if r < forest_radius
        scalefactor = 16 * 10 * (tanh((r + forest_radius * 0.9) / smoothing_disance) - tanh((r - forest_radius * 0.9) / smoothing_disance))/2
        push!(kelps, GiantKelp(; grid, base_x = x, base_y = y, base_z = -8.0, scalefactor = scalefactor, architecture = arch))
    end
end

particle_struct = StructArray(kelps);=#

x = Float64[]
y = Float64[]
x0 = Float64[]
y0 = Float64[]
sf = Float64[]

for x in xnodes(Center, grid), y in ynodes(Center, grid)
    r = sqrt((x - Lx/2)^2 + (y - Ly/2)^2)
    if r < forest_radius
        scalefactor = 16 * 10 * (tanh((r + forest_radius * 0.9) / smoothing_disance) - tanh((r - forest_radius * 0.9) / smoothing_disance))/2
        push!(x, x)
        push!(y, y)
        push!(x0, x)
        push!(y0, y)
        push!(sf, scalefactor)
    end
end

n_kelp = length(x)

node_positions = zeros(Float64, n_kelp, 8, 3)
blade_areas = zeros(Float64, n_kelp, 8)
drag_fields = Vector{Field}()
for i in 1:n_kelp
    node_positions[i, :, :] = GiantKelpDynamics.x⃗₀(8, 8, 0.6, 2.5)
    blade_areas[i, :] = 0.1 .* [i*50/8 for i in 1:8]
    push!(drag_fields, CenterField(grid))
end

particle_struct = StructArray{GiantKelp}(arch_array(arch, x), # x
                                         arch_array(arch, y), # y
                                         arch_array(arch, -8.0 * ones(n_kelp)), # z
                                         arch_array(arch, x), # x0
                                         arch_array(arch, y), # y0
                                         arch_array(arch, -8.0 * ones(n_kelp)), # z0
                                         arch_array(arch, sf), # scalefactor
                                         arch_array(arch, node_positions), # node_positions
                                         arch_array(arch, zeros(Float64, n_kelp, 8, 3)), # node_velocities
                                         arch_array(arch, 0.6 * ones(Float64, n_kelp, 8)), # relaxed length
                                         arch_array(arch, 0.03 * ones(Float64, n_kelp, 8)), # stipe radii
                                         arch_array(arch, blade_areas), # blade area
                                         arch_array(arch, 0.05 * ones(Float64, n_kelp, 8)), # pneumatocysts volume
                                         arch_array(arch, 0.5 * ones(Float64, n_kelp, 8)), # effective radii
                                         arch_array(arch, zeros(Float64, n_kelp, 8, 3)), # accelerations
                                         arch_array(arch, zeros(Float64, n_kelp, 8, 3)), # old velocities
                                         arch_array(arch, zeros(Float64, n_kelp, 8, 3)), # old accelerations
                                         arch_array(arch, zeros(Float64, n_kelp, 8, 3)), # drag force
                                         arch_array(arch, drag_fields)) # drag fiedls

@inline guassian_smoothing(r, rᵉ) = 1.0#exp(-(r)^2/(2*rᵉ^2))/sqrt(2*π*rᵉ^2)

particles = LagrangianParticles(particle_struct; 
                                dynamics = kelp_dynamics!, 
                                parameters = (k = 10 ^ 5, 
                                              α = 1.41, 
                                              ρₒ = 1026.0, 
                                              ρₐ = 1.225, 
                                              g = 9.81, 
                                              Cᵈˢ = 1.0, 
                                              Cᵈᵇ= 0.4 * 12 ^ -0.485, 
                                              Cᵃ = 3.0,
                                              drag_smoothing = guassian_smoothing,
                                              n_nodes = 8,
                                              kᵈ = 10 ^ 4)) # for a linear spring system, to be non-oscillatory we need kᵈ>√k

drag_nodes = CenterField(grid)

@inline tidal_forcing(x, y, z, t, params) = - params.Aᵤ * params.ω * sin(params.ω * (t - params.t_central) - params.ϕᵤ) - params.Aᵥ * params.ω * cos(params.ω * (t - params.t_central) - params.ϕᵥ)

u_bcs = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0.0))
v_bcs = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0.0))
w_bcs = FieldBoundaryConditions(bottom = OpenBoundaryCondition(0.0))
#Aᵤ = 7.00e-2, Aᵥ = 4.68e-2, ϕᵤ = 1.038, ϕᵥ = 3.80, t_central = 1.58e7, ω = 6.76e-5
model = NonhydrostaticModel(; grid,
                              advection = CenteredSecondOrder(),
                              timestepper = :RungeKutta3,
                              closure = AnisotropicMinimumDissipation(),
                              forcing = (u = Forcing(tidal_forcing, parameters = (Aᵤ = 0.25, Aᵥ = 0.0, ϕᵤ = 0.0, ϕᵥ = 0.0, t_central = 0.0, ω = 6.76e-5)), ),
                              boundary_conditions = (u = u_bcs, v = v_bcs, w = w_bcs),
                              particles = particles,
                              auxiliary_fields = (; drag_nodes))

uᵢ(x, y, z) = 0.25 * cos(6.76e-5 * 0.0)
#vᵢ(x, y, z) = 4.68e-2 * cos(- 6.76e-5 * 1.58e7 - 3.80)

set!(model, u = uᵢ)

filepath = "forest_no_coupling"

simulation = Simulation(model, Δt = 1.0, stop_time = 1year)

#wizard = TimeStepWizard(cfl = 0.5, max_change = 1.1, diffusive_cfl = 0.5)
#simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

progress_message(sim) = @printf("Iteration: %07d, time: %s, Δt: %s, max(|u|) = %.1e ms⁻¹, min(|u|) = %.1e ms⁻¹, wall time: %s\n",
                                    iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                    maximum(abs, sim.model.velocities.u), minimum(abs, sim.model.velocities.u), prettytime(sim.run_wall_time))
    
simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(20))

simulation.output_writers[:profiles] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                         filename = "$filepath.jld2",
                         schedule = TimeInterval(1minute),
                         overwrite_existing = true)

function store_particles!(sim)
    jldopen("$(filepath)_particles.jld2", "a+") do file
        file["x⃗/$(sim.model.clock.time)"] = sim.model.particles.properties.node_positions
    end
end

simulation.callbacks[:save_particles] = Callback(store_particles!, TimeInterval(1minute))
simulation.stop_time = 30
run!(simulation)

simulation.callbacks[:drag_water] = Callback(drag_water!; callsite = TendencyCallsite())
simulation.stop_time = 3minutes
run!(simulation)

#=
file = jldopen("$(filepath)_particles.jld2")
times = keys(file["x⃗"])
x⃗ = zeros(length(times), 8, 3)
for (i, t) in enumerate(times)
    x⃗[i, :, :] = file["x⃗/$t"][1].x⃗
end
close(file)
u = FieldTimeSeries("$filepath.jld2", "u") .- u₀;

using CairoMakie

gx = [Lx/Nx:Lx/Nx:Lx;]
gy = [Lx/Nx:Lx/Nx:Ly;]
gz = [-Lz:Lx/Nx:Lx/Nx;]

fig = Figure(resolution = (2000, 2000/(Lx/Lz)))
ax  = Axis(fig[1, 1]; xlabel="x (m)", ylabel="z (m)", title="t=$(prettytime(0))", limits=(0, Lx, -Lz, 0))

# animation settings
nframes = length(times)
framerate = floor(Int, nframes/30)
frame_iterator = 1:nframes

n = Observable(1)
x = @lift (x⃗ .+ 5.0)[$n, :, 1]
y = @lift (x⃗ .+ 2.0)[$n, :, 2]
z = @lift (x⃗ .- 8.0)[$n, :, 3]


u_plt = @lift u[1:Nx, floor(Int, Ny/2), 1:Nz, $n]
uₘ = maximum(abs, u[1:Nx, floor(Int, Ny/2), 1:Nz, :])
hmu = heatmap!(ax, gx, gz, u_plt, colormap=:vik, colorrange=(-uₘ, uₘ))
Colorbar(fig[1, 2], hmu)
plot!(ax, x, z)
for i=1:8
    lines!(ax, x⃗[:, i, 1].+5.0, x⃗[:, i, 3].-8.0)
end
CairoMakie.record(fig, "$(filepath)_vertical_slice.mp4", frame_iterator; framerate = framerate) do i
    print("$i" * " \r")
    n[] = i
    ax.title = "t=$(prettytime(parse(Float64, times[i])))"
end


n = Observable(1)
fig = Figure(resolution = (2000, 2000/(.5*Lx/Ly)))
ax_u  = Axis(fig[1, 1]; xlabel="x (m)", ylabel="y (m)")
x = @lift (x⃗.+5)[$n, :, 1]
y = @lift (x⃗.+2)[$n, :, 2]
z = @lift (x⃗.-8)[$n, :, 3]
colors = lift(getcol, z)
u_plt = @lift u[1:Nx, 1:Ny, Nz, $n]

uₘ = maximum(abs, u[1:Nx, 1:Ny, Nz, :])
hmu = heatmap!(ax_u, gx, gy, u_plt, colormap=:vik, colorrange=(-uₘ, uₘ))
Colorbar(fig[1, 2], hmu)

plot!(ax_u, x, y, color=colors)
CairoMakie.record(fig, "$(filepath)_horizontal_u.mp4", frame_iterator; framerate = framerate) do i
    n[] = i
    msg = string("Plotting frame ", i, " of ", nframes)
    print(msg * " \r")
    ax_u.title = "t=$(prettytime(parse(Float64, times[i])))"
end

fig = Figure()

z_xy = zeros(Float64, Nx, Ny)
y_xz = ones(Float64, Nx, Nz).*2
x_yz = zeros(Float64, Ny, Nz)

ax = Axis3(fig[1, 1], aspect=(1, .5*Ly/Lx, Lz/Lx), limits=(0, Lx, Ly/2, Ly, -Lz, 0))

u_plt_top = @lift u[1:Nx, 16:Ny, Nz, $n]
sl1 = surface!(ax, gx, gy, z_xy; color=u_plt_top, colormap=:vik, colorrange=(0.0, 2.0))

u_plt_west = @lift u[1, 16:Ny, 1:Nz, $n]
sl2 = surface!(ax, x_yz, gy[16:end], gz; color=u_plt_west, colormap=:vik, colorrange=(0.0, 2.0))

u_plt_center = @lift u[1:Nx, 16, 1:Nz, $n]
sl3 = surface!(ax, gx, y_xz, gz; color=u_plt_center, colormap=:vik, colorrange=(0.0, 2.0))

x = @lift (x⃗.+5)[$n, :, 1]
y = @lift (x⃗.+2)[$n, :, 2]
z = @lift (x⃗.-8)[$n, :, 3]

plot!(ax, x, y, z)
GLMakie.record(fig, "$(filepath)_3d_plot.mp4", frame_iterator; framerate = framerate) do i
    n[] = i
    msg = string("Plotting frame ", i, " of ", nframes)
    print(msg * " \r")
    ax.title = "t=$(prettytime(parse(Float64, times[i])))"
end
=#