using Oceananigans, StructArrays, Printf, JLD2
using Oceananigans.Units: minutes, minute, hour, hours, day

include("macrosystis_dynamics.jl")

# ## Setup grid 
Lx, Ly, Lz = 12, 4, 8
Nx, Ny, Nz = 4 .*(Lx, Ly, Lz)
grid = RectilinearGrid(size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz), topology=(Bounded, Periodic, Bounded))

# ## Setup kelp particles
n_kelp = 1
n_seg = 8

x₀ = repeat([grid.xᶜᵃᵃ[1]], n_kelp)
y₀ = repeat([grid.yᵃᶜᵃ[8]], n_kelp)
z₀ = repeat([grid.zᵃᵃᶜ[1]], n_kelp)

l⃗₀₀ = repeat([1.0], n_seg)
r⃗ˢ₀ = repeat([0.03], n_seg)
n⃗ᵇ₀ = repeat([10], n_seg)
n⃗ᵇ₀[end] = 50
A⃗ᵇ₀ = repeat([0.1], n_seg)
x⃗₀ = zeros(n_seg, 3)
x⃗₀[:, 3] .= [sum(l⃗₀₀[1:k]) for k=1:n_seg]
x⃗₀[:, 3][x⃗₀[:, 3] .- 8 .> 0] .= 0.0
u⃗₀ = zeros(n_seg, 3)
V⃗ᵖ₀ = repeat([0.002], n_seg) # currently assuming pneumatocysts have density 500kg/m³

individuals_nodes = Nodes(x⃗₀, u⃗₀, l⃗₀₀, r⃗ˢ₀, n⃗ᵇ₀, A⃗ᵇ₀, V⃗ᵖ₀, zeros(n_seg, 3), zeros(n_seg, 3), zeros(n_seg, 3), zeros(n_seg, 3))

# this only works here where there is one particle (i.e. assumes all kelp have same base position)
nodes = repeat([individuals_nodes], n_kelp)

kelp_particles = StructArray{GiantKelp}((x₀, y₀, z₀, zeros(n_kelp), zeros(n_kelp), zeros(n_kelp), nodes))

# here I am assuming the blades behave as streamers with aspect ratio approx 12 https://arc.aiaa.org/doi/pdf/10.2514/1.9754

particles = LagrangianParticles(kelp_particles; dynamics=kelp_dynamics!, parameters=(k = 10^5, α = 1.41, ρₒ = 1026.0, ρₐ = 1.225, g = 9.81, Cᵈˢ = 1.0, Cᵈᵇ=0.4*12^(-0.485), Cᵃ = 3.0)) #α=1.41, k=2e5 ish for Utter/Denny

u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(0.0), bottom = ValueBoundaryCondition(0.0), east = OpenBoundaryCondition(0.15), west = OpenBoundaryCondition(0.15))
v_bcs = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0.0))
w_bcs = FieldBoundaryConditions(bottom = OpenBoundaryCondition(0.0))

model = NonhydrostaticModel(; grid,
                                advection = WENO(),
                                timestepper = :RungeKutta3,
                                closure = ScalarDiffusivity(ν=1e-4, κ=1e-4),
                                boundary_conditions = (u=u_bcs, v=v_bcs, w=w_bcs),
                                particles=particles)
set!(model, u=0.15)

simulation = Simulation(model, Δt=0.1, stop_time=2minutes)
wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=0.25, diffusive_cfl=0.5)
#simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))
progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|u|) = %.1e ms⁻¹, wall time: %s\n",
                                    iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                    maximum(abs, sim.model.velocities.u), prettytime(sim.run_wall_time))
    
simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(20))

filepath = "kelp"

simulation.output_writers[:profiles] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                         filename = "$filepath.jld2",
                         indices = (:, grid.Ny/2, :),
                         schedule = TimeInterval(10),
                         overwrite_existing = true)

function store_particles!(sim)
    jldopen("$(filepath)_particles.jld2", "a+") do file
        file["x⃗/$(sim.model.clock.time)"] = sim.model.particles.properties.nodes
    end
end

simulation.callbacks[:save_particles] = Callback(store_particles!)
run!(simulation)

file = jldopen("$(filepath)_particles.jld2")
times = keys(file["x⃗"])
x⃗ = zeros(length(times), n_seg, 3)
for (i, t) in enumerate(times)
    x⃗[i, :, :] = file["x⃗/$t"][1].x⃗
end

using GLMakie
fig = Figure(resolution = (1000*maximum(x⃗[:, :, 1])/maximum(x⃗[:, :, 3]), 500))
ax  = Axis(fig[1, 1]; limits=((0, maximum(x⃗[:, :, 1])), (0, maximum(x⃗[:, :, 3]))), xlabel="x (m)", ylabel="z (m)", title="t=$(prettytime(0))", aspect = AxisAspect(maximum(x⃗[:, :, 1])/maximum(x⃗[:, :, 3])))

# animation settings
nframes = length(times)
framerate = floor(Int, nframes/30)
frame_iterator = 1:nframes

record(fig, "nodes.mp4", frame_iterator; framerate = framerate) do i
    msg = string("Plotting frame ", i, " of ", nframes)
    print(msg * " \r")
    if !(i==1)
        plot!(ax, x⃗[(i-1), :, 1], x⃗[(i-1), :, 3]; color=:white)
    end
    plot!(ax, x⃗[i, :, 1], x⃗[i, :, 3])
    ax.title = "t=$(prettytime(parse(Float64, times[i])))"
end