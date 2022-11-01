using Oceananigans, StructArrays, Printf, JLD2
using Oceananigans.Units: minutes, minute, hour, hours, day

include("macrosystis_dynamics.jl")

# ## Setup grid 
grid = RectilinearGrid(size=(16, 16, 16), extent=(32, 32, 8), topology=(Periodic, Periodic, Bounded))

# ## Setup kelp particles
n_kelp = 1
n_seg = 8

x₀ = repeat([grid.xᶜᵃᵃ[8]], n_kelp)
y₀ = repeat([grid.yᵃᶜᵃ[8]], n_kelp)
z₀ = repeat([grid.zᵃᵃᶜ[1]], n_kelp)

l⃗₀₀ = repeat([1.0], n_seg)
x⃗₀ = zeros(n_seg, 3)
x⃗₀[:, 3] .= [sum(l⃗₀₀[1:k]) for k=1:n_seg]
x⃗₀[:, 3][x⃗₀[:, 3] .- 8 .> 0] .= 0.0
u⃗₀ = zeros(n_seg, 3)
ρ⃗₀ = repeat([800.0], n_seg)
V⃗₀ = repeat([1*0.2*0.02], n_seg)
A⃗ᶠ₀ = repeat([1*0.2], n_seg)
A⃗ᶜ₀ = repeat([π*0.02^2], n_seg)

individuals_nodes = Nodes(x⃗₀, u⃗₀, ρ⃗₀, V⃗₀, A⃗ᶠ₀, A⃗ᶜ₀, l⃗₀₀, zeros(n_seg, 3), zeros(n_seg, 3), zeros(n_seg, 3))

# this only works here where there is one particle (i.e. assumes all kelp have same base position)
nodes = repeat([individuals_nodes], n_kelp)

kelp_particles = StructArray{GiantKelp}((x₀, y₀, z₀, zeros(n_kelp), zeros(n_kelp), zeros(n_kelp), zeros(n_kelp), zeros(n_kelp), zeros(n_kelp), nodes))

particles = LagrangianParticles(kelp_particles; dynamics=dynamics!, parameters=(k = 10^5, α = 1.41, ρₒ = 1026.0, ρₐ = 1.225, g = 9.81, Cᵈ = 1.0, Cᵃ = 3.0)) #α=1.41, k=2e5 ish for Utter/Denny

# ## Setup model
u₁₀ = 10.0    # m s⁻¹, average wind velocity 10 meters above the ocean
cᴰ = 2.5e-3 # dimensionless drag coefficient
ρₐ = 1.225  # kg m⁻³, average density of air at sea-level
ρₒ = 1026.0 # kg m⁻³, average density at the surface of the world ocean

Qᵘ = 0#- ρₐ / ρₒ * cᴰ * u₁₀ * abs(u₁₀) # m² s⁻²

u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))

model = NonhydrostaticModel(; grid,
                            advection = WENO(),
                            timestepper = :RungeKutta3,
                            closure = ScalarDiffusivity(ν=1e-4, κ=1e-4),
                            tracers = :A,
                            boundary_conditions = (u=u_bcs, ),
                            particles = particles)

Ξ(z) = randn() * z / model.grid.Lz * (1 + z / model.grid.Lz) 
uᵢ(x, y, z) = sqrt(abs(Qᵘ)) * 1e-3 * Ξ(z)
set!(model, u=0.15, w=uᵢ, v=uᵢ, A=1.0)

simulation = Simulation(model, Δt=0.2, stop_time=2minutes)

progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|u|) = %.1e ms⁻¹, wall time: %s\n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                maximum(abs, sim.model.velocities.u), prettytime(sim.run_wall_time))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(20))

filepath = "kelp"

simulation.output_writers[:profiles] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                     filename = "$filepath.jld2",
                     indices = (:, grid.Ny/2, :),
                     schedule = TimeInterval(1minute),
                     overwrite_existing = true)


function store_particles!(sim)
    jldopen("$(filepath)_particles.jld2", "a+") do file
        file["x⃗/$(sim.model.clock.time)"] = sim.model.particles.properties.nodes
    end
end

simulation.callbacks[:save_particles] = Callback(store_particles!)
run!(simulation)

file = jldopen("kelp_particles.jld2")
times = keys(file["x⃗"])
x⃗ = zeros(length(times), n_seg, 3)
for (i, t) in enumerate(times)
    x⃗[i, :, :] = file["x⃗/$t"][1].x⃗
end

using GLMakie
fig = Figure(resolution = (1000, 500))
ax  = Axis(fig[1, 1]; limits=((0, maximum(x⃗[:, :, 1])), (0, maximum(x⃗[:, :, 3]))), xlabel="x (m)", ylabel="z (m)", title="t=$(prettytime(0))", aspect = AxisAspect(maximum(x⃗[:, :, 1])/maximum(x⃗[:, :, 3])))

# animation settings
nframes = length(times)
framerate = floor(Int, nframes/2minutes)
frame_iterator = range(1, nframes)

record(fig, "nodes.mp4", frame_iterator; framerate = framerate) do i
    msg = string("Plotting frame ", i, " of ", nframes)
    print(msg * " \r")
    if !(i==1)
        plot!(ax, x⃗[i-1, :, 1], x⃗[i-1, :, 3]; color=:white)
    end
    plot!(ax, x⃗[i, :, 1], x⃗[i, :, 3])
    ax.title = "t=$(prettytime(parse(Float64, times[i])))"
end