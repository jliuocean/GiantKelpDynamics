using Oceananigans, StructArrays, Printf, JLD2, Statistics, CUDA
using Oceananigans.Units: minutes, minute, hour, hours, day
using Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_w_from_continuity!

using GiantKelpDynamics

arch = CUDA.has_cuda_gpu() ? Oceananigans.GPU() : Oceananigans.CPU()

# ## Setup grid 
Lx, Ly, Lz = 32, 4, 8
Nx, Ny, Nz = (floor(Int, Lx / 4), floor(Int, Ly / 4), Lz)
grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz), topology=(Periodic, Periodic, Bounded))

# ## Setup kelp particles

n_nodes = 16

kelps = GiantKelp(; grid, 
                    segment_unstretched_length = [0.2 + (n - 1) * 0.1 for n in 1:n_nodes],
                    initial_stretch = 1,
                    number_nodes = n_nodes, 
                    base_x = [5.0], base_y = [2.0], base_z = [-8.0], 
                    architecture = arch, 
                    max_Δt = 0.15,
                    drag_fields = false,
                    parameters = (k = 10 ^ 5, 
                                    α = 1.41, 
                                    ρₒ = 1026.0, 
                                    ρₐ = 1.225, 
                                    g = 9.81, 
                                    Cᵈˢ = 1.0, 
                                    Cᵈᵇ= 0.4 * 12 ^ -0.485, 
                                    Cᵃ = 3.0,
                                    n_nodes = n_nodes,
                                    τ = 1.0,
                                    kᵈ = 500))

#=
BenchmarkTools.Trial: 192 samples with 1 evaluation.
 Range (min … max):  18.662 ms … 209.061 ms  ┊ GC (min … max):  0.00% … 89.43%
 Time  (median):     20.956 ms               ┊ GC (median):     0.00%
 Time  (mean ± σ):   26.475 ms ±  25.683 ms  ┊ GC (mean ± σ):  13.85% ± 12.74%

  █▅▃                                                           
  ███▇▄▅▅▅▁▅▆▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▅ ▄
  18.7 ms       Histogram: log(frequency) by time       192 ms <

 Memory estimate: 38.16 MiB, allocs estimate: 312142.

BenchmarkTools.Trial: 169 samples with 1 evaluation.
 Range (min … max):  22.554 ms … 250.832 ms  ┊ GC (min … max):  0.00% … 85.37%
 Time  (median):     25.013 ms               ┊ GC (median):     0.00%
 Time  (mean ± σ):   29.602 ms ±  28.367 ms  ┊ GC (mean ± σ):  12.58% ± 11.67%

  █▄                                                            
  ██▇▅▅▁▄▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄ ▄
  22.6 ms       Histogram: log(frequency) by time       232 ms <

 Memory estimate: 56.07 MiB, allocs estimate: 558627.
=#

u₀ = 0.2

u_bcs = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0.0))
v_bcs = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0.0))
w_bcs = FieldBoundaryConditions(bottom = OpenBoundaryCondition(0.0))

#u_forcing_func(args...) = 1e-5
#u_forcing = Forcing(u_forcing_func, discrete_form=true)
@inline mask_rel_U(x, y, z) = ifelse(x < 3, 1, 0)
@inline relax_U(x, y, z, t, u, params) = 10 * mask_rel_U(x, y, z) * (params.u₀*(1 + 0.01*rand()) - u)
U_forcing = Forcing(relax_U, field_dependencies = (:u, ), parameters = (; u₀))

@inline mask_rel_V(x, y, z) = ifelse(x < 3, 1, 0)
@inline relax_V(x, y, z, t, v) = - 10 * mask_rel_V(x, y, z) * v
V_forcing = Forcing(relax_V, field_dependencies = (:v, ))

@inline mask_rel_W(x, y, z) = ifelse(x < 3, 1, 0)
@inline relax_W(x, y, z, t, w) = - 10 * mask_rel_W(x, y, z) * w
W_forcing = Forcing(relax_W, field_dependencies = (:w, ))

@inline N_background(x, y, z, t) = tanh(-2 * z / 8)
#N_relax = Relaxation(; rate = 1/10, target = N_background)
mask_N(x, y, z) = ifelse(x < 3, 1, 0)
@inline relax_N(x, y, z, t, N) = 1 * mask_N(x, y, z) * (N_background(x, y, z, t) - N)
N_forcing = Forcing(relax_N, field_dependencies = (:N, ))

drag_u, drag_v, drag_w = DiscreteDrags(; particles = kelps)

model = NonhydrostaticModel(; grid,
                              advection = WENO(grid),
                              timestepper = :RungeKutta3,
                              closure = ScalarDiffusivity(κ = 1e-2, ν = 1e-2),
                              boundary_conditions = (u=u_bcs, v=v_bcs, w=w_bcs),
                              forcing = (u = drag_u, v = drag_v, w = drag_w, N = N_forcing),#v = V_forcing, w = W_forcing, N = N_forcing),
                              particles = kelps,
                              tracers = :N)

uᵢ(x, y, z) = u₀*(1 + randn()*0.01)
vᵢ(x, y, z) = u₀*randn()*0.1
Nᵢ(x, y, z) = N_background(x, y, z, 0.0)
set!(model, u=uᵢ, v=vᵢ, w=vᵢ, N=Nᵢ)

filepath = "very_low_res"

simulation = Simulation(model, Δt=0.2, stop_time=3minute)

wizard = TimeStepWizard(cfl=0.5, max_change=1.1, diffusive_cfl=0.5)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(1))

progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|u|) = %.1e ms⁻¹, wall time: %s\n",
                                    iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                    maximum(abs, sim.model.velocities.u), prettytime(sim.run_wall_time))
    
simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(20))


simulation.output_writers[:profiles] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                         filename = "$filepath.jld2",
                         schedule = TimeInterval(1),
                         overwrite_existing = true)

function store_particles!(sim)
    jldopen("$(filepath)_particles.jld2", "a+") do file
        file["x⃗/$(sim.model.clock.time)"] = sim.model.particles.properties.positions
    end
end
simulation.callbacks[:save_particles] = Callback(store_particles!, TimeInterval(1))

# transient response to the kelp starting to move 

simulation.stop_time = 3minutes
simulation.Δt = 0.1
#run!(simulation)

#=
file = jldopen("$(filepath)_particles.jld2")
times = keys(file["x⃗"])
x⃗ = zeros(length(times), n_nodes, 3)
for (i, t) in enumerate(times)
    x⃗[i, :, :] = file["x⃗/$t"][1]
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
CairoMakie.record(fig, "$(filepath)_3d_plot.mp4", frame_iterator; framerate = framerate) do i
    n[] = i
    msg = string("Plotting frame ", i, " of ", nframes)
    print(msg * " \r")
    ax.title = "t=$(prettytime(parse(Float64, times[i])))"
end
=#