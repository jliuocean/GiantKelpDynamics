using Oceananigans, StructArrays, Printf, JLD2, Statistics, CUDA
using Oceananigans.Units: minutes, minute, hour, hours, day

include("macrosystis_dynamics.jl")

arch = CUDA.has_cuda_gpu() ? Oceananigans.GPU() : Oceananigans.CPU()

if arch == Oceananigans.CPU()
    adapt_array(array) = Array(array)
elseif arch == Oceananigans.GPU()
    adapt_array(array) = CuArray(array)
else
    error("Incorrect arch type")
end

# ## Setup grid 
Lx, Ly, Lz = 64, 4, 8
Nx, Ny, Nz = 8 .*(Lx, Ly, Lz)
grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz), topology=(Periodic, Periodic, Bounded))

# ## Setup kelp particles

x⃗₀ = zeros(8, 3)
zᶜ = 0.0
for i=1:8
    global zᶜ += 12/8
    if zᶜ - 8.0 < 0
        x⃗₀[i, :] = [0.0, 0.0, zᶜ]
    else
        x⃗₀[i, :] = [zᶜ - 8, 0.0, 8.0]
    end
end

x⃗₀[:, 2] .= 0.0

n⃗ᵇ = [i*50/8 for i = 1:8]
A⃗ᵇ = repeat([0.1], 8)

nodes = Nodes(adapt_array.((x⃗₀, 
                            zeros(8, 3), 
                            repeat([.6], 8), 
                            repeat([0.03], 8), 
                            n⃗ᵇ, #repeat([1], 8), 
                            A⃗ᵇ, #repeat([1], 8), 
                            repeat([0.05], 8), 
                            repeat([0.5], 8), 
                            zeros(8, 3), 
                            zeros(8, 3), 
                            zeros(8, 3), 
                            zeros(8, 3)))...)

particle_struct = StructArray{GiantKelp}(adapt_array.(([5.0], [2.0], [-8.0], [5.0], [2.0], [-8.0], [nodes])))

@inline guassian_smoothing(r, rᵉ) = 1.0#exp(-(r)^2/(2*rᵉ^2))/sqrt(2*π*rᵉ^2)

particles = LagrangianParticles(particle_struct; 
                            dynamics = kelp_dynamics!, 
                            parameters = (k = 10^5, 
                                          α = 1.41, 
                                          ρₒ = 1026.0, 
                                          ρₐ = 1.225, 
                                          g = 9.81, 
                                          Cᵈˢ = 1.0, 
                                          Cᵈᵇ=0.4*12^(-0.485), 
                                          Cᵃ = 3.0,
                                          drag_smoothing = guassian_smoothing,
                                          n_nodes = 8))

u₀=1.0

u_bcs = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0.0))
v_bcs = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0.0))
w_bcs = FieldBoundaryConditions(bottom = OpenBoundaryCondition(0.0))

#u_forcing_func(args...) = 1e-5
#u_forcing = Forcing(u_forcing_func, discrete_form=true)
@inline mask_rel_U(x, y, z) = ifelse(x < 3, 1, 0)
@inline relax_U(x, y, z, t, u, params) = 10 * mask_rel_U(x, y, z) * (params.u₀ - u)
U_forcing = Forcing(relax_U, field_dependencies = (:u, ), parameters = (; u₀))

@inline mask_rel_V(x, y, z) = ifelse(x < 3, 1, 0)
@inline relax_V(x, y, z, t, v) = - 10 * mask_rel_V(x, y, z) * v
V_forcing = Forcing(relax_V, field_dependencies = (:v, ))

@inline mask_rel_W(x, y, z) = ifelse(x < 3, 1, 0)
@inline relax_W(x, y, z, t, w) = - 10 * mask_rel_W(x, y, z) * w
W_forcing = Forcing(relax_W, field_dependencies = (:w, ))


N_background(x, y, z, t) = tanh(2*z/8)
N_relax = Relaxation(; 1/10, target = N_background)

drag_nodes = CenterField(grid)

model = NonhydrostaticModel(; grid,
                              advection = WENO(),
                              timestepper = :RungeKutta3,
                              closure = nothing,#ScalarDiffusivity(ν=1e-4, κ=1e-4),
                              boundary_conditions = (u=u_bcs, v=v_bcs, w=w_bcs),
                              forcing = (u = U_forcing, v = V_forcing, w = W_forcing, N = N_relax),
                              particles = particles,
                              auxiliary_fields = (; drag_nodes),
                              tracers = :N)

uᵢ(x, y, z) = u₀*randn()*0.01
vᵢ(x, y, z) = u₀*randn()*0.01
set!(model, u=uᵢ, v=vᵢ, w=vᵢ)

filepath = "tracer"

simulation = Simulation(model, Δt=0.1, stop_time=3minute)

simulation.callbacks[:drag_water] = Callback(drag_water!; callsite = TendencyCallsite())

wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=0.25, diffusive_cfl=0.5)
#simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|u|) = %.1e ms⁻¹, wall time: %s\n",
                                    iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                    maximum(abs, sim.model.velocities.u), prettytime(sim.run_wall_time))
    
simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(20))


simulation.output_writers[:profiles] =
    JLD2OutputWriter(model, model.velocities,
                         filename = "$filepath.jld2",
                         schedule = IterationInterval(1),
                         overwrite_existing = true)

function store_particles!(sim)
    jldopen("$(filepath)_particles.jld2", "a+") do file
        file["x⃗/$(sim.model.clock.time)"] = sim.model.particles.properties.nodes
    end
end

simulation.callbacks[:save_particles] = Callback(store_particles!, IterationInterval(1))
#=
using GLMakie

fig = Figure(resolution = (2000, 2000/(Lx/Ly)))
ax_u  = Axis(fig[1, 1]; xlabel="x (m)", ylabel="y (m)")
getcol(z) = RGBAf.(0, 0, 0, (8 .+ z)./10 .+.2)
function plot(sim)
    model = sim.model
    uₘ = maximum(abs, model.velocities.u[:, :, Nz] .- u₀)
    hm = heatmap!(ax_u, grid.xᶜᵃᵃ[1:Nx], grid.yᵃᶜᵃ[1:Ny], model.velocities.u[1:Nx, 1:Ny, Nz] .- u₀, colorrange=(-uₘ, uₘ), colormap=:vik)
    plt = plot!(ax_u, nodes.x⃗[:, 1] .+ 5, nodes.x⃗[:, 2] .+ 2, color=getcol.(nodes.x⃗[:, 3] .- 8))
    fig
end

simulation.callbacks[:plot] = Callback(plot, IterationInterval(20))=#
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

gx = [0.125:0.125:Lx;]
gy = [0.125:0.125:Ly;]
gz = [-Lz:0.125:0.125;]

fig = Figure(resolution = (2000, 2000/(Lx/Lz)))
ax  = Axis(fig[1, 1]; xlabel="x (m)", ylabel="z (m)", title="t=$(prettytime(0))")

# animation settings
nframes = length(times)
framerate = floor(Int, nframes/30)
frame_iterator = 1:nframes

n = Observable(1)
x = @lift (x⃗ .+ 5.0)[$n, :, 1]
y = @lift (x⃗ .+ 2.0)[$n, :, 2]
z = @lift (x⃗ .- 8.0)[$n, :, 3]


u_plt = @lift u[1:Nx, 16, 1:Nz, $n]
uₘ = maximum(abs, u[1:Nx, 16, 1:Nz, :])
hmu = heatmap!(ax, gx, gz, u_plt, colormap=:vik, colorrange=(-uₘ, uₘ))
Colorbar(fig[1, 2], hmu)
plot!(ax, x, z)
for i=1:8
    lines!(ax, x⃗[:, i, 1].+5.0, x⃗[:, i, 3].-8.0)
end
CairoMakie.record(fig, "$(filepath)_nodes_dragging.mp4", frame_iterator; framerate = framerate) do i
    print("$i" * " \r")
    n[] = i
    ax.title = "t=$(prettytime(parse(Float64, times[i])))"
end
=#

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