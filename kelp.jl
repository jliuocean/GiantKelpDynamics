using Oceananigans, StructArrays, Printf, JLD2, Statistics
using Oceananigans.Units: minutes, minute, hour, hours, day

include("macrosystis_dynamics.jl")

# ## Setup grid 
Lx, Ly, Lz = 32, 4, 8
Nx, Ny, Nz = 4 .*(Lx, Ly, Lz)
grid = RectilinearGrid(size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz), topology=(Periodic, Periodic, Bounded))

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

nodes = Nodes(x⃗₀, 
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
              zeros(8, 3))

particle_struct = StructArray{GiantKelp}(([12.0], [2.0], [-8.0], [12.0], [2.0], [-8.0], [nodes]))

@inline guassian_smoothing(r, rᵉ) = exp(-(3*r)^2/(2*rᵉ^2))/sqrt(2*π*rᵉ^2)

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

u_bcs = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0.0))
v_bcs = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0.0))
w_bcs = FieldBoundaryConditions(bottom = OpenBoundaryCondition(0.0))

u_forcing_func(args...) = 1e-5
u_forcing = Forcing(u_forcing_func, discrete_form=true)

drag_nodes = CenterField(grid)

model = NonhydrostaticModel(; grid,
                                advection = WENO(),
                                timestepper = :RungeKutta3,
                                closure = ScalarDiffusivity(ν=1e-4, κ=1e-4),
                                boundary_conditions = (u=u_bcs, v=v_bcs, w=w_bcs),
                                forcing = (u = u_forcing, ),
                                particles = particles,
                                auxiliary_fields = (; drag_nodes))
set!(model, u=u₀)

filepath = "dragging_new"

simulation = Simulation(model, Δt=0.1, stop_time=30.0)

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
                         schedule = IterationInterval(50),
                         overwrite_existing = true)

function store_particles!(sim)
    jldopen("$(filepath)_particles.jld2", "a+") do file
        file["x⃗/$(sim.model.clock.time)"] = sim.model.particles.properties.nodes
    end
end

simulation.callbacks[:save_particles] = Callback(store_particles!, IterationInterval(50))
#=run!(simulation)

file = jldopen("$(filepath)_particles.jld2")
times = keys(file["x⃗"])
x⃗ = zeros(length(times), 8, 3)
for (i, t) in enumerate(times)
    x⃗[i, :, :] = file["x⃗/$t"][1].x⃗
end

using GLMakie

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
record(fig, "nodes_dragging.mp4", frame_iterator; framerate = framerate) do i
    print("$i" * " \r")
    n[] = i
    ax.title = "t=$(prettytime(parse(Float64, times[i])))"
end

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
record(fig, "horizontal_u.mp4", frame_iterator; framerate = framerate) do i
    n[] = i
    msg = string("Plotting frame ", i, " of ", nframes)
    print(msg * " \r")
    ax_u.title = "t=$(prettytime(parse(Float64, times[i])))"
end

=#