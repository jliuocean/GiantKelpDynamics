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
# ## Setup grid 
Lx, Ly, Lz = 1kilometers, 1kilometers, 8
Nx, Ny, Nz = 256, 256, 8
grid = RectilinearGrid(arch, FT; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz), topology=(Periodic, Periodic, Bounded))

# ## Setup kelp particles

forest_radius = 100
smoothing_disance = 3.0

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
        scalefactor = base_scaling * (tanh((r + forest_radius * 0.9) / smoothing_disance) - tanh((r - forest_radius * 0.9) / smoothing_disance))/2
        push!(xs, x)
        push!(ys, y)
        push!(sf, scalefactor)
    end
end

number_nodes = 3
segment_unstretched_length = [3, .5, .5]

kelps = GiantKelp(; grid, 
                    number_nodes, 
                    segment_unstretched_length, 
                    base_x = xs, base_y = ys, base_z = -8.0 * ones(length(xs)), 
                    scalefactor = sf, 
                    architecture = arch, 
                    max_Δt = 0.15, # TODO: experiment with this now I know where weird transient comes from
                    drag_fields = false,
                    parameters = (k = 10 ^ 5, 
                                  α = 1.41, 
                                  ρₒ = 1026.0, 
                                  ρₐ = 1.225, 
                                  g = 9.81, 
                                  Cᵈˢ = 1.0, 
                                  Cᵈᵇ= 0.4 * 12 ^ -0.485, 
                                  Cᵃ = 3.0,
                                  n_nodes = number_nodes,
                                  τ = 1.0,
                                  kᵈ = 500))

@inline tidal_forcing(x, y, z, t, params) = - params.Aᵤ * params.ω * sin(params.ω * (t - params.t_central) - params.ϕᵤ) - params.Aᵥ * params.ω * cos(params.ω * (t - params.t_central) - params.ϕᵥ)

drag_set = DiscreteDragSet(; grid, xy_smudge_distance = max(1, floor(Int, (node_density/2) - 1)))

# I think this ω gives a period of 1 day but it should be 12 hours?
u_forcing = (Forcing(tidal_forcing, parameters = (Aᵤ = 0.25, Aᵥ = 0.0, ϕᵤ = -π/2, ϕᵥ = 0.0, t_central = 6hours, ω = 1.41e-4)), 
             drag_set.u)
v_forcing = drag_set.v
w_forcing = drag_set.w

u_bcs = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0.0))
v_bcs = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0.0))
w_bcs = FieldBoundaryConditions(bottom = OpenBoundaryCondition(0.0))
#Aᵤ = 7.00e-2, Aᵥ = 4.68e-2, ϕᵤ = 1.038, ϕᵥ = 3.80, t_central = 1.58e7, ω = 6.76e-5
model = NonhydrostaticModel(; grid,
                              advection = CenteredSecondOrder(),
                              timestepper = :RungeKutta3,
                              closure = AnisotropicMinimumDissipation(),
                              forcing = (u = u_forcing, v = v_forcing, w = w_forcing),
                              boundary_conditions = (u = u_bcs, v = v_bcs, w = w_bcs),
                              coriolis = FPlane(; latitude = 34.5),
                              particles = kelps)

uᵢ(x, y, z) = 0.25 * cos(u_forcing[1].parameters.ω * 0.0 - u_forcing[1].parameters.ϕᵤ) + 0.25 * 0.01 * (rand() - 0.5)
#vᵢ(x, y, z) = 4.68e-2 * cos(- 6.76e-5 * 1.58e7 - 3.80)

set!(model, u = uᵢ)

Δt₀ = 0.5
# initialise kelp positions_ijk
kelp_dynamics!(kelps, model, Δt₀)

filepath = "forest_vertical_smooshing"

simulation = Simulation(model, Δt = Δt₀, stop_time = 1year)

simulation.callbacks[:update_drag_fields] = Callback(drag_set; callsite = TendencyCallsite())

wizard = TimeStepWizard(cfl = 0.8, max_change = 1.1, min_change = 0.8)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

progress_message(sim) = @printf("Iteration: %07d, time: %s, Δt: %s, max(|u|) = %.1e ms⁻¹, min(|u|) = %.1e ms⁻¹, wall time: %s\n",
                                 iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                 maximum(abs, sim.model.velocities.u), minimum(abs, sim.model.velocities.u), prettytime(sim.run_wall_time))
    
simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(1minute))

simulation.output_writers[:profiles] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                         filename = "$filepath.jld2",
                         schedule = TimeInterval(1minute),
                         overwrite_existing = true)

function store_particles!(sim)
    jldopen("$(filepath)_particles.jld2", "a+") do file
        file["x⃗/$(sim.model.clock.time)"] = sim.model.particles.properties.positions
    end
end

simulation.callbacks[:save_particles] = Callback(store_particles!, TimeInterval(1minute))

simulation.stop_time = 10days

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
    ax.title = "t=$(prettytime(parse(FT, times[i])))"
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
    ax_u.title = "t=$(prettytime(parse(FT, times[i])))"
end

fig = Figure()

z_xy = zeros(FT, Nx, Ny)
y_xz = ones(FT, Nx, Nz).*2
x_yz = zeros(FT, Ny, Nz)

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
    ax.title = "t=$(prettytime(parse(FT, times[i])))"
end
=#
