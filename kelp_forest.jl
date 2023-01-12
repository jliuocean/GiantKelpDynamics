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
grid = RectilinearGrid(arch, FT;
                       size=(Nx, Ny, Nz), 
                       extent=(Lx, Ly, Lz),
                       topology=(Periodic, Bounded, Bounded))

# ## Setup kelp particles

forest_radius = 100
smoothing_disance = 3.0

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
        scalefactor = base_scaling * (tanh((r + forest_radius * 0.9) / smoothing_disance) - tanh((r - forest_radius * 0.9) / smoothing_disance))/2
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
                                  Cᵈᵇ= 0.3 * 12 ^ -0.485, 
                                  Cᵃ = 3.0,
                                  n_nodes = number_nodes,
                                  τ = 1.0,
                                  kᵈ = 500))

@inline tidal_forcing(x, y, z, t, params) = - params.Aᵤ * params.ω * sin(params.ω * (t - params.t_central) - params.ϕᵤ) - params.Aᵥ * params.ω * cos(params.ω * (t - params.t_central) - params.ϕᵥ)

drag_set = DiscreteDragSet(; grid)

# I think this ω gives a period of 1 day but it should be 12 hours?
u_forcing = (Forcing(tidal_forcing, parameters = (Aᵤ = 0.15, Aᵥ = 0.05, ϕᵤ = -π/2, ϕᵥ = -π, t_central = 0, ω = 1.41e-4)), 
             drag_set.u)

v_forcing = (Forcing(tidal_forcing, parameters = (Aᵤ = 0.05, Aᵥ = 0.15, ϕᵤ = -π, ϕᵥ = -π/2, t_central = 0, ω = 1.41e-4)),
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

uᵢ(x, y, z) = 0.15 * cos(π/2) + 0.15 * (rand() - 0.5) * 2 * 0.01
vᵢ(x, y, z) = 0.05 * cos(π) * (1 + (rand() - 0.5) * 2 * 0.01)

set!(model, u = uᵢ, v = vᵢ)

Δt₀ = 0.5
# initialise kelp positions_ijk
kelp_dynamics!(kelps, model, Δt₀)

filepath = "forest_depth_avg_velocity_new"

simulation = Simulation(model, Δt = Δt₀, stop_time = 1year)

simulation.callbacks[:update_drag_fields] = Callback(drag_set; callsite = TendencyCallsite())

wizard = TimeStepWizard(cfl = 0.8, max_change = 1.1, min_change = 0.8)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

progress_message(sim) = @printf("Iteration: %07d, time: %s, Δt: %s, max(|u|) = %.1e ms⁻¹, min(|u|) = %.1e ms⁻¹, wall time: %s\n",
                                 iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                 maximum(abs, sim.model.velocities.u), minimum(abs, sim.model.velocities.u), prettytime(sim.run_wall_time))
    
simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(5minute))

simulation.output_writers[:profiles] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                         filename = "$filepath.jld2",
                         schedule = TimeInterval(5minute),
                         overwrite_existing = true)

function store_particles!(sim)
    jldopen("$(filepath)_particles.jld2", "a+") do file
        file["x⃗/$(sim.model.clock.time)"] = sim.model.particles.properties.positions
    end
end

simulation.callbacks[:save_particles] = Callback(store_particles!, TimeInterval(5minute))

simulation.output_writers[:checkpointer] = Checkpointer(model, schedule = TimeInterval(1hour), overwrite_existing = true)

simulation.stop_time = 10days

run!(simulation, pickup = false)

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

using EasyFit

u₁ = u[76, 128, Nz, :]
u₃ = u[110, 128, Nz, :]

u₁_pos = u₁[u₁ .>= 0]
u₁_neg = u₁[u₁ .< 0]

u₃_pos = u₃[u₁ .>= 0]
u₃_neg = u₃[u₁ .< 0]

u₁³_pos = fitlinear(u₁_pos, u₃_pos)
u₁³_neg = fitlinear(u₁_neg, u₃_neg)

fig = Figure(resolution = (1600, 1600))
ax = Axis(fig[1, 1])

scatter!(ax, u₁, u₃)
umax = max(maximum(abs, u₁), maximum(abs, u₃))
lines!(ax, [-umax, umax], [-umax, umax], color=:black, linestyle=:dot)
lines!(ax, u₁³_pos.x, u₁³_pos.y, color=:black, label = "$(u₁³_pos.b) + $(u₁³_pos.a) x, r² = $(u₁³_pos.R^2)")
lines!(ax, u₁³_neg.x, u₁³_neg.y, color=:black, label = "$(u₁³_neg.b) + $(u₁³_neg.a) x, r² = $(u₁³_neg.R^2)")
axislegend(ax, position = :lt)

save("$(filepath)_west.png", fig)

u₁₃ = u[180, 128, Nz, :]
u₁₁ = u[148, 128, Nz, :]

u₁₃_pos = u₁₃[u₁₃ .>= 0]
u₁₃_neg = u₁₃[u₁₃ .< 0]

u₁₁_pos = u₁₁[u₁₃ .>= 0]
u₁₁_neg = u₁₁[u₁₃ .< 0]

u₁₃¹¹_pos = fitlinear(u₁₃_pos, u₁₁_pos)
u₁₃¹¹_neg = fitlinear(u₁₃_neg, u₁₁_neg)

fig = Figure(resolution = (1600, 1600))
ax = Axis(fig[1, 1])

scatter!(ax, u₁₃, u₁₁)
umax = max(maximum(abs, u₁₃), maximum(abs, u₁₁))
lines!(ax, [-umax, umax], [-umax, umax], color=:black, linestyle=:dot)
lines!(ax, u₁₃¹¹_pos.x, u₁₃¹¹_pos.y, color=:black, label = "$(u₁₃¹¹_pos.b) + $(u₁₃¹¹_pos.a) x, r² = $(u₁₃¹¹_pos.R^2)")
lines!(ax, u₁₃¹¹_neg.x, u₁₃¹¹_neg.y, color=:black, label = "$(u₁₃¹¹_neg.b) + $(u₁₃¹¹_neg.a) x, r² = $(u₁₃¹¹_neg.R^2)")
axislegend(ax, position = :lt)

save("$(filepath)_east.png", fig)

return "$(filepath)_west.png", "$(filepath)_east.png"