using Oceananigans, StructArrays, Printf, JLD2
using Oceananigans.Units: minutes, minute, hour, hours, day

include("macrosystis_dynamics.jl")

# ## Setup grid 
grid = RectilinearGrid(size=(16, 16, 16), extent=(32, 32, 32), topology=(Periodic, Periodic, Bounded))

# ## Setup kelp particles
n_kelp = 1
n_seg = 8

x₀ = repeat([grid.xᶜᵃᵃ[8]], n_kelp)
y₀ = repeat([grid.yᵃᶜᵃ[8]], n_kelp)
z₀ = repeat([grid.zᵃᵃᶜ[1]], n_kelp)

l⃗₀₀ = repeat([1.0], n_seg)
x⃗₀ = zeros(n_seg, 3)
x⃗₀[:, 3] .= [sum(l⃗₀₀[1:k]) for k=1:n_seg]
u⃗₀ = zeros(n_seg, 3)
ρ⃗₀ = repeat([800.0], n_seg)
V⃗₀ = repeat([1*0.2*0.02], n_seg)
A⃗ᶠ₀ = repeat([1*0.2], n_seg)
A⃗ᶜ₀ = repeat([π*0.02^2], n_seg)

individuals_nodes = Nodes(x⃗₀, u⃗₀, ρ⃗₀, V⃗₀, A⃗ᶠ₀, A⃗ᶜ₀, l⃗₀₀, zeros(n_seg, 3), zeros(n_seg, 3), zeros(n_seg, 3))

# this only works here where there is one particle (i.e. assumes all kelp have same base position)
nodes = repeat([individuals_nodes], n_kelp)

kelp_particles = StructArray{GiantKelp}((x₀, y₀, z₀, zeros(n_kelp), zeros(n_kelp), zeros(n_kelp), zeros(n_kelp), zeros(n_kelp), zeros(n_kelp), nodes))

particles = LagrangianParticles(kelp_particles; dynamics=dynamics!, parameters=(k = 10^5, α = 1.41, ρₒ = 1026.0, g = 9.81, Cᵈ = 1.0)) #α=1.41, k=2e5 ish for Utter/Denny

# ## Setup model
u₁₀ = 10    # m s⁻¹, average wind velocity 10 meters above the ocean
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

simulation = Simulation(model, Δt=0.1, stop_time=1day)

wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=10minutes, diffusive_cfl=0.5)
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
                     schedule = TimeInterval(1minute),
                     overwrite_existing = true)


function store_particles!(sim)
    jldopen("$(filepath)_particles.jld2", "a+") do file
        file["x⃗/$(sim.model.clock.time)"] = sim.model.particles.properties.nodes
    end
end

simulation.callbacks[:save_particles] = Callback(store_particles!)
run!(simulation)

using GLMakie

u = FieldTimeSeries("$filepath.jld2", "u")
v = FieldTimeSeries("$filepath.jld2", "v")
w = FieldTimeSeries("$filepath.jld2", "w")
A = FieldTimeSeries("$filepath.jld2", "A")

xu, yu, zu = nodes(u)
xv, yv, zv = nodes(v)
xw, yw, zw = nodes(w)
xA, yA, zA = nodes(A)

times =u.times
intro = searchsortedfirst(times, 1minutes)

n = Observable(intro)

uₙ = @lift interior(u[$n],  :, 1, :)
vₙ = @lift interior(v[$n],  :, 1, :)
wₙ = @lift interior(w[$n],  :, 1, :)
Aₙ = @lift interior(A[$n],  :, 1, :)

fig = Figure(resolution = (1000, 500))

axis_kwargs = (xlabel="x (m)",
               ylabel="z (m)",
               aspect = AxisAspect(grid.Lx/grid.Lz),
               limits = ((0, grid.Lx), (-grid.Lz, 0)))

ax_u  = Axis(fig[2, 1]; title = "u", axis_kwargs...)
ax_v  = Axis(fig[2, 3]; title = "v", axis_kwargs...)
ax_w  = Axis(fig[3, 1]; title = "w", axis_kwargs...)
ax_A  = Axis(fig[3, 3]; title = "A", axis_kwargs...)

title = @lift @sprintf("t = %s", prettytime(times[$n]))

ulim = (min(0, minimum(u)), maximum(u))
vlim = (minimum(v), maximum(v))
wlim = (minimum(w), maximum(w))
Alim = (minimum(A), maximum(A))

hm_u = heatmap!(ax_u, xu, zu, uₙ; colormap = :batlow, colorrange = ulim)
Colorbar(fig[2, 2], hm_u; label = "m s⁻¹")

hm_v = heatmap!(ax_v, xv, zv, vₙ; colormap = :balance, colorrange = vlim)
Colorbar(fig[2, 4], hm_v; label = "m s⁻¹")

hm_w = heatmap!(ax_w, xw, zw, wₙ; colormap = :balance, colorrange = wlim)
Colorbar(fig[3, 2], hm_w; label = "m s⁻¹")

hm_w = heatmap!(ax_A, xA, zA, Aₙ; colormap = :balance, colorrange = wlim)
Colorbar(fig[3, 4], hm_w; label = "m⁻³")

fig[1, 1:4] = Label(fig, title, textsize=24, tellwidth=false)

frames = intro:length(times)

record(fig, filepath * ".mp4", frames, framerate=16) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")
    n[] = i
end