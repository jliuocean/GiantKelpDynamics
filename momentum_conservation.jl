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
Lx, Ly, Lz = 24, 4, 8
Nx, Ny, Nz = 4 .*(Lx, Ly, Lz)
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

n⃗ᵇ = [20 for i = 1:8]
A⃗ᵇ = repeat([0.1], 8)

nodes = Nodes(adapt_array.((x⃗₀, 
                            zeros(8, 3), 
                            repeat([.8*12/8], 8), 
                            repeat([0.03], 8), 
                            n⃗ᵇ, #repeat([1], 8), 
                            A⃗ᵇ, #repeat([1], 8), 
                            repeat([0.003], 8), 
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

u₀=0.2

drag_nodes = CenterField(grid)

model = NonhydrostaticModel(; grid,
                                advection = WENO(),
                                timestepper = :RungeKutta3,
                                closure = ScalarDiffusivity(ν=1e-4, κ=1e-4),
                                particles = particles,
                                auxiliary_fields = (; drag_nodes))

uᵢ(x, y, z) = u₀
set!(model, u=uᵢ)

filepath = "momentum_conservation"

simulation = Simulation(model, Δt=0.5, stop_time=1minute)

simulation.callbacks[:drag_water] = Callback(drag_water!; callsite = TendencyCallsite())

progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|u|) = %.1e ms⁻¹, wall time: %s\n",
                                    iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                    maximum(abs, sim.model.velocities.u), prettytime(sim.run_wall_time))
    
simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(20))

∫adV = Integral(model.timestepper.Gⁿ.u, dims=(1, 2, 3))

simulation.output_writers[:profiles] =
    JLD2OutputWriter(model, merge(model.velocities, (; ∫adV)),
                         filename = "$filepath.jld2",
                         schedule = IterationInterval(1),
                         overwrite_existing = true)

function store_particles!(sim)
    jldopen("$(filepath)_particles.jld2", "a+") do file
        file["x⃗/$(sim.model.clock.time)"] = sim.model.particles.properties.nodes
    end
end

simulation.callbacks[:save_particles] = Callback(store_particles!, IterationInterval(1))

run!(simulation)
#=
file = jldopen("$(filepath)_particles.jld2")
times = keys(file["x⃗"])
x⃗ = zeros(length(times), 8, 3)
for (i, t) in enumerate(times)
    x⃗[i, :, :] = file["x⃗/$t"][1].x⃗
end

using CairoMakie

fig = Figure(resolution = (1000*1.1*maximum(x⃗[:, :, 1])/maximum(x⃗[:, :, 3]), 1000))
ax  = Axis(fig[1, 1]; limits=((min(0, minimum(x⃗[:, :, 1])), 1.1*maximum(x⃗[:, :, 1])), (min(0, minimum(x⃗[:, :, 3])), 1.1*maximum(x⃗[:, :, 3]))), xlabel="x (m)", ylabel="z (m)", title="t=$(prettytime(0))", aspect = AxisAspect(maximum(x⃗[:, :, 1])/maximum(x⃗[:, :, 3])))

# animation settings
nframes = length(times)
framerate = floor(Int, nframes/30)
frame_iterator = 1:nframes

n = Observable(1)
x = @lift x⃗[$n, :, 1]
y = @lift x⃗[$n, :, 2]
z = @lift x⃗[$n, :, 3]


plot!(ax, x, z)
for i=1:8
    lines!(ax, x⃗[:, i, 1], x⃗[:, i, 3])
end
CairoMakie.record(fig, "nodes_dragging.mp4", frame_iterator; framerate = framerate) do i
    print("$i" * " \r")
    n[] = i
    ax.title = "t=$(prettytime(parse(Float64, times[i])))"
end
=#
u = FieldTimeSeries("$filepath.jld2", "u") .- u₀;

n = Observable(1)
fig = Figure(resolution = (2000, 2000/(grid.Lx/grid.Ly)))
ax_u  = Axis(fig[1, 1]; aspect = AxisAspect(grid.Lx/grid.Ly), xlabel="x (m)", ylabel="y (m)")
x = @lift (x⃗.+particles.properties.x[1])[$n, :, 1]
y = @lift (x⃗.+particles.properties.y[1])[$n, :, 2]
z = @lift (x⃗.+particles.properties.z[1])[$n, :, 3]
getcol(z) = RGBAf.(0, 0, 0, (8 .+ z)./10 .+.2)
colors = lift(getcol, z)
u_plt = @lift u[1:Nx, 1:Ny, Nz, $n]

uₘ = maximum(abs, u[1:Nx, 1:Ny, Nz, :])
hmu = heatmap!(ax_u, grid.xᶠᵃᵃ[1:Nx], grid.yᵃᶜᵃ[1:Ny], u_plt, colormap=:vik, colorrange=(-uₘ, uₘ))
Colorbar(fig[1, 2], hmu)

plot!(ax_u, x, y, color=colors)
CairoMakie.record(fig, "horizontal_u.mp4", frame_iterator; framerate = framerate) do i
    n[] = i
    msg = string("Plotting frame ", i, " of ", nframes)
    print(msg * " \r")
    ax_u.title = "t=$(prettytime(parse(Float64, times[i])))"
end
