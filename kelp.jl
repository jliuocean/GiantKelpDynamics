using Oceananigans, StructArrays, Printf, JLD2, Statistics
using Oceananigans.Units: minutes, minute, hour, hours, day

include("macrosystis_dynamics.jl")

# ## Setup grid 
Lx, Ly, Lz = 64, 8, 8
Nx, Ny, Nz = 2 .*(Lx, Ly, Lz)
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

n⃗ᵇ = [i*50/8 for i = 1:8]
A⃗ᵇ = repeat([0.1], 8)

nodes = Nodes(x⃗₀, 
              zeros(8, 3), 
              repeat([.75*12/8], 8), 
              repeat([0.03], 8), 
              n⃗ᵇ, #repeat([1], 8), 
              A⃗ᵇ, #repeat([1], 8), 
              repeat([0.003], 8), 
              repeat([0.5], 8), 
              zeros(8, 3), 
              zeros(8, 3), 
              zeros(8, 3), 
              zeros(8, 3))

particle_struct = StructArray{GiantKelp}(([12.0], [4.0], [-8.0], [12.0], [4.0], [-8.0], [nodes]))

function guassian_smoothing(r, z, rᵉ)
    if z>0
        r = sqrt(r^2 + z^2)
        return 0.0#exp(-(7*r)^2/(2*rᵉ^2))/(2*sqrt(2*π*rᵉ^2))
    else
        return exp(-(7*r)^2/(2*rᵉ^2))/sqrt(2*π*rᵉ^2)
    end
end

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
                                          drag_smoothing = guassian_smoothing))

u₀=0.2

u_bcs = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0.0))
v_bcs = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0.0))
w_bcs = FieldBoundaryConditions(bottom = OpenBoundaryCondition(0.0))

background_U(x, y, z, t) = ifelse(x <= 8, u₀, 0.0)
mask_rel(x, y, z) = ifelse(x < 5, 1, 0)
relax_U = Relaxation(1/2, mask_rel, background_U)

background_perp(x, y, z, t) = 0.0
relax_perp = Relaxation(1/2, mask_rel, background_perp)


drag_nodes = CenterField(grid)

model = NonhydrostaticModel(; grid,
                                advection = WENO(),
                                timestepper = :RungeKutta3,
                                #closure = ScalarDiffusivity(ν=1e-4, κ=1e-4),
                                #boundary_conditions = (u=u_bcs, v=v_bcs, w=w_bcs),
                                #forcing = (u = relax_U, v = relax_perp, w = relax_perp),
                                particles = particles,
                                auxiliary_fields = (; drag_nodes))
set!(model, u=u₀)

simulation = Simulation(model, Δt=0.1, stop_time=3minutes)

simulation.callbacks[:drag_water] = Callback(drag_water!; callsite = TendencyCallsite())

wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=0.25, diffusive_cfl=0.5)
#simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|u|) = %.1e ms⁻¹, wall time: %s\n",
                                    iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                    maximum(abs, sim.model.velocities.u), prettytime(sim.run_wall_time))
    
simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(20))

filepath = "dragging"

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

simulation.callbacks[:save_particles] = Callback(store_particles!)
run!(simulation)


file = jldopen("$(filepath)_particles.jld2")
times = keys(file["x⃗"])
x⃗ = zeros(length(times), 8, 3)
for (i, t) in enumerate(times)
    x⃗[i, :, :] = file["x⃗/$t"][1].x⃗
end

using GLMakie

fig = Figure(resolution = (1000*maximum(x⃗[:, :, 1])/maximum(x⃗[:, :, 3]), 1000))
ax  = Axis(fig[1, 1]; limits=((min(0, minimum(x⃗[:, :, 1])), maximum(x⃗[:, :, 1])), (min(0, minimum(x⃗[:, :, 3])), maximum(x⃗[:, :, 3]))), xlabel="x (m)", ylabel="z (m)", title="t=$(prettytime(0))", aspect = AxisAspect(maximum(x⃗[:, :, 1])/maximum(x⃗[:, :, 3])))

# animation settings
nframes = 1000
framerate = floor(Int, nframes/30)
frame_iterator = 1:nframes

n = Observable(1)
x = @lift x⃗[$n, :, 1]
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

#=
fig = Figure(resolution = (500*grid.Lx/grid.Lz, 500))
ax_u  = Axis(fig[1, 1]; title = "u", aspect = AxisAspect(grid.Lx/grid.Lz), xlabel="x (m)", ylabel="z (m)")
u_plt = model.velocities.u[1:Nx, floor(Int, Ny/2), 1:Nz, end].-u₀
uₘ = maximum(abs, u_plt)
hmu = heatmap!(ax_u, grid.xᶠᵃᵃ[1:Nx], grid.zᵃᵃᶜ[1:Nz], u_plt, colormap=:vik, colorrange=(-uₘ, uₘ))
Colorbar(fig[1, 2], hmu)
scatter!(ax_u, model.particles.properties.nodes[1].x⃗[:, 1].+model.particles.properties.x[1], model.particles.properties.nodes[1].x⃗[:, 3].+model.particles.properties.z[1], color=:black, markersize=20)
scatter!(ax_u, [model.particles.properties.x[1]], [model.particles.properties.z[1]], color=:black, markersize=20)
save("vertical_u_slice.png", fig)

pNz = Nz
fig = Figure(resolution = (500*grid.Lx/grid.Ly, 500))
ax_u  = Axis(fig[1, 1]; title = "u", aspect = AxisAspect(grid.Lx/grid.Ly), xlabel="x (m)", ylabel="y (m)")
u_plt = model.velocities.u[1:Nx, 1:Ny, pNz, end].-u₀
uₘ = maximum(abs, u_plt)
hmu = heatmap!(ax_u, grid.xᶠᵃᵃ[1:Nx], grid.yᵃᶜᵃ[1:Ny], u_plt, colormap=:vik, colorrange=(-uₘ, uₘ))
Colorbar(fig[1, 2], hmu)
scatter!(ax_u, model.particles.properties.nodes[1].x⃗[:, 1].+model.particles.properties.x[1], model.particles.properties.nodes[1].x⃗[:, 2].+model.particles.properties.y[1], color=:black, markersize=20)
scatter!(ax_u, [model.particles.properties.x[1]], [model.particles.properties.y[1]], color=:black, markersize=20)
save("horizontal_u_slice.png", fig)

fig = Figure(resolution = (1000, 500))
ax_u  = Axis(fig[1, 1]; title = "u", xlabel="x (m)", ylabel="z (m)")
lines!(ax_u, model.velocities.u[modf(fractional_x_index(25, Face(), grid))[2]+1, floor(Int, Ny/2), 1:Nz]./mean(model.velocities.u[modf(fractional_x_index(25, Face(), grid))[2]+1, floor(Int, Ny/2), 1:Nz]), grid.zᵃᵃᶜ[1:Nz])
ax_v  = Axis(fig[2, 1]; title = "Horizontal profile at x=20m, z=0m", xlabel="y (m)", ylabel="u-u₀ (m/s)")
lines!(ax_v, grid.yᵃᶜᵃ[1:Ny],  model.velocities.u[modf(fractional_x_index(20, Face(), grid))[2]+1, 1:Ny, Nz].-u₀)
save("u_profiles.png", fig)

u = FieldTimeSeries("$filepath.jld2", "u")
u_plt = mean(u[1:Nx, 1:Ny, 1:Nz, :], dims=2)[:, 1, :, :].-u₀
uₘ = maximum(abs, u_plt)

fig = Figure(resolution = (500*grid.Lx/grid.Lz, 500))
ax  = Axis(fig[1, 1]; aspect = AxisAspect(grid.Lx/grid.Lz), xlabel="x (m)", ylabel="z (m)")

# animation settings
nframes = length(times)
framerate = floor(Int, nframes/30)
frame_iterator = 1:nframes

hmu = heatmap!(ax, grid.xᶠᵃᵃ[1:Nx], grid.zᵃᵃᶜ[1:Nz], u_plt[1:Nx, 1:Nz, 1], colormap=:vik, colorrange=(-uₘ, uₘ))
Colorbar(fig[1, 2], hmu)
scatter!(ax, [particles.properties.x[1]], [particles.properties.z[1]], color=:black, markersize=20)

record(fig, "dragging.mp4", frame_iterator; framerate = framerate) do i
    msg = string("Plotting frame ", i, " of ", nframes)
    print(msg * " \r")

    hmu = heatmap!(ax, grid.xᶠᵃᵃ[1:Nx], grid.zᵃᵃᶜ[1:Nz], u_plt[1:Nx, 1:Nz, i], colormap=:vik, colorrange=(-uₘ, uₘ))
    if !(i==1)
        scatter!(ax, x⃗[i-1, :, 1].+particles.properties.x[1], x⃗[i-1, :, 3].+particles.properties.z[1], color=:white, markersize=20)
    end
    scatter!(ax, x⃗[i, :, 1].+particles.properties.x[1], x⃗[i, :, 3].+particles.properties.z[1], color=:black, markersize=20)
    ax.title = "t=$(prettytime(parse(Float64, times[i])))"
end


fig = Figure(resolution = (500*grid.Lx/grid.Ly, 500))
ax_u  = Axis(fig[1, 1]; aspect = AxisAspect(grid.Lx/grid.Ly), xlabel="x (m)", ylabel="y (m)")
u_plt = u[1:Nx, 1:Ny, Nz, :] .-u₀
uₘ = maximum(abs, u_plt)
hmu = heatmap!(ax_u, grid.xᶠᵃᵃ[1:Nx], grid.yᵃᶜᵃ[1:Ny], u_plt[1:Nx, 1:Ny, 1], colormap=:vik, colorrange=(-uₘ, uₘ))
Colorbar(fig[1, 2], hmu)
scatter!(ax_u, x⃗[1, :, 1].+particles.properties.x[1], x⃗[1, :, 2].+particles.properties.y[1], color=:black, markersize=20)
scatter!(ax_u, [particles.properties.x[1]], [particles.properties.y[1]], color=:black, markersize=20)

record(fig, "horizontal_u.mp4", frame_iterator; framerate = framerate) do i
    msg = string("Plotting frame ", i, " of ", nframes)
    print(msg * " \r")
    hmu = heatmap!(ax_u, grid.xᶠᵃᵃ[1:Nx], grid.yᵃᶜᵃ[1:Ny], u_plt[1:Nx, 1:Ny, i], colormap=:vik, colorrange=(-uₘ, uₘ))
    scatter!(ax_u, x⃗[i, :, 1].+particles.properties.x[1], x⃗[i, :, 2].+particles.properties.y[1], color=:black, markersize=20)
    ax_u.title = "t=$(prettytime(parse(Float64, times[i])))"
end=#