using Oceananigans, StructArrays, Printf, JLD2
using Oceananigans.Units: minutes, minute, hour, hours, day

include("macrosystis_dynamics.jl")

# ## Setup grid 
Lx, Ly, Lz = 36, 3, 8
Nx, Ny, Nz = 8 .*(Lx, Ly, Lz)
grid = RectilinearGrid(size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz), topology=(Periodic, Periodic, Bounded))

# ## Setup kelp particles
n_kelp = 1
n_seg = 8
Lₖₑₗₚ = 1.5*Lz
l₀ = Lₖₑₗₚ/n_seg

x = repeat([10.0], n_kelp)
y = repeat([Ly/2], n_kelp)
z = repeat([0.125-Lz], n_kelp)

const x₀ = repeat([10.0], n_kelp)
const y₀ = repeat([Ly/2], n_kelp)
const z₀ = repeat([0.125-Lz], n_kelp)

l⃗₀₀ = repeat([l₀], n_seg)
r⃗ˢ₀ = repeat([0.03], n_seg)
r⃗ᵉ₀ = repeat([0.5], n_seg)
n⃗ᵇ₀ = repeat([0], n_seg)
n⃗ᵇ₀[end] = 50
A⃗ᵇ₀ = repeat([0.1], n_seg)
x⃗₀ = zeros(n_seg, 3)
zᶜ = 0.0
for i=1:n_seg
    global zᶜ += l⃗₀₀[i]
    if zᶜ+z₀[1] < 0
        x⃗₀[i, :] = [0.0, 0.0, zᶜ]
    else
        x⃗₀[i, :] = [zᶜ+z₀[1], 0.0, -z₀[1]]
    end
end
u⃗₀ = zeros(n_seg, 3)
V⃗ᵖ₀ = repeat([0.002], n_seg) # currently assuming pneumatocysts have density 500kg/m³

individuals_nodes = Nodes(x⃗₀, u⃗₀, l⃗₀₀, r⃗ˢ₀, n⃗ᵇ₀, A⃗ᵇ₀, V⃗ᵖ₀, r⃗ᵉ₀, zeros(n_seg, 3), zeros(n_seg, 3), zeros(n_seg, 3), zeros(n_seg, 3))

# this only works here where there is one particle (i.e. assumes all kelp have same base position)
nodes = repeat([individuals_nodes], n_kelp)

kelp_particles = StructArray{GiantKelp}((x, y, z, x₀, y₀, z₀, nodes))

# here I am assuming the blades behave as streamers with aspect ratio approx 12 https://arc.aiaa.org/doi/pdf/10.2514/1.9754

particles = LagrangianParticles(kelp_particles; dynamics=kelp_dynamics!, parameters=(k = 10^5, α = 1.41, ρₒ = 1026.0, ρₐ = 1.225, g = 9.81, Cᵈˢ = 1.0, Cᵈᵇ=0.4*12^(-0.485), Cᵃ = 3.0)) #α=1.41, k=2e5 ish for Utter/Denny

u₀=0.2

u_bcs = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0.0))
v_bcs = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0.0))
w_bcs = FieldBoundaryConditions(bottom = OpenBoundaryCondition(0.0))

background_U(x, y, z, t) = ifelse(x <= 5, u₀, 0.0)
mask_U(x, y, z) = ifelse(x < 5, 1, 0)
relax_U = Relaxation(1/2, mask_U, background_U)

model = NonhydrostaticModel(; grid,
                                advection = WENO(),
                                timestepper = :RungeKutta3,
                                closure = ScalarDiffusivity(ν=1e-4, κ=1e-4),
                                #boundary_conditions = (u=u_bcs, v=v_bcs, w=w_bcs),
                                forcing = (u = relax_U, ),
                                particles=particles)
set!(model, u=u₀)

simulation = Simulation(model, Δt=0.5, stop_time=2minutes)

simulation.callbacks[:drag_water] = Callback(drag_water!; callsite = TendencyCallsite())

wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=0.25, diffusive_cfl=0.5)
#simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))
progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|u|) = %.1e ms⁻¹, wall time: %s\n",
                                    iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                    maximum(abs, sim.model.velocities.u), prettytime(sim.run_wall_time))
    
simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(20))

filepath = "kelp_dragging"

simulation.output_writers[:profiles] =
    JLD2OutputWriter(model, model.velocities,
                         filename = "$filepath.jld2",
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
fig = Figure(resolution = (1000*maximum(x⃗[:, :, 1])/maximum(x⃗[:, :, 3]), 1000))
ax  = Axis(fig[1, 1]; limits=((min(0, minimum(x⃗[:, :, 1])), maximum(x⃗[:, :, 1])), (min(0, minimum(x⃗[:, :, 3])), maximum(x⃗[:, :, 3]))), xlabel="x (m)", ylabel="z (m)", title="t=$(prettytime(0))", aspect = AxisAspect(maximum(x⃗[:, :, 1])/maximum(x⃗[:, :, 3])))

# animation settings
nframes = length(times)
framerate = floor(Int, nframes/30)
frame_iterator = 1:nframes

record(fig, "nodes_dragging.mp4", frame_iterator; framerate = framerate) do i
    msg = string("Plotting frame ", i, " of ", nframes)
    print(msg * " \r")
    if !(i==1)
        plot!(ax, x⃗[(i-1), :, 1], x⃗[(i-1), :, 3]; color=:white)
    end
    plot!(ax, x⃗[i, :, 1], x⃗[i, :, 3])
    ax.title = "t=$(prettytime(parse(Float64, times[i])))"
end

fig = Figure(resolution = (500*grid.Lx/grid.Lz, 500))
ax_u  = Axis(fig[1, 1]; title = "u", aspect = AxisAspect(grid.Lx/grid.Lz), xlabel="x (m)", ylabel="z (m)")
u_plt = model.velocities.u[1:Nx, floor(Int, Ny/2), 1:Nz, end].-u₀
uₘ = maximum(abs, u_plt)
hmu = heatmap!(ax_u, grid.xᶠᵃᵃ[1:Nx], grid.zᵃᵃᶜ[1:Nz], u_plt, colormap=:vik, colorrange=(-uₘ, uₘ))
Colorbar(fig[1, 2], hmu)
scatter!(ax_u, model.particles.properties.nodes[1].x⃗[:, 1].+model.particles.properties.x[1], model.particles.properties.nodes[1].x⃗[:, 3].+model.particles.properties.z[1], color=:black, markersize=20)
scatter!(ax_u, [model.particles.properties.x[1]], [model.particles.properties.z[1]], color=:black, markersize=20)
save("u_velocity_slice.png", fig)

pNz = Nz-1
fig = Figure(resolution = (500*grid.Lx/grid.Ly, 500))
ax_u  = Axis(fig[1, 1]; title = "u", aspect = AxisAspect(grid.Lx/grid.Ly), xlabel="x (m)", ylabel="y (m)")
u_plt = model.velocities.u[1:Nx, 1:Ny, pNz, end].-u₀
uₘ = maximum(abs, u_plt)
hmu = heatmap!(ax_u, grid.xᶠᵃᵃ[1:Nx], grid.yᵃᶜᵃ[1:Ny], u_plt, colormap=:vik, colorrange=(-uₘ, uₘ))
Colorbar(fig[1, 2], hmu)
scatter!(ax_u, model.particles.properties.nodes[1].x⃗[:, 1].+model.particles.properties.x[1], model.particles.properties.nodes[1].x⃗[:, 2].+model.particles.properties.y[1], color=:black, markersize=20)
scatter!(ax_u, [model.particles.properties.x[1]], [model.particles.properties.y[1]], color=:black, markersize=20)
save("u_velocity_profile.png", fig)