using Oceananigans, Printf
using Oceananigans.Units: minutes, minute, hour, hours, day

Nx, Ny, Nz = 16, 16, 16
grid = RectilinearGrid(size=(Nx, Ny, Nz), extent=(32, 32, 32), topology=(Periodic, Bounded, Bounded))

u₁₀ = 10    # m s⁻¹, average wind velocity 10 meters above the ocean
cᴰ = 2.5e-3 # dimensionless drag coefficient
ρₐ = 1.225  # kg m⁻³, average density of air at sea-level
ρₒ = 1026.0 # kg m⁻³, average density at the surface of the world ocean

Qᵘ = - ρₐ / ρₒ * cᴰ * u₁₀ * abs(u₁₀) # m² s⁻²

u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ), bottom = ValueBoundaryCondition(0.0), north = ValueBoundaryCondition(0.0), south = ValueBoundaryCondition(0.0))
no_slip = FieldBoundaryConditions(ValueBoundaryCondition(0.0))

model = NonhydrostaticModel(; grid,
                            timestepper = :RungeKutta3,
                            closure = ScalarDiffusivity(ν=1e-2, κ=1e-2),
                            tracers = :A,
                            boundary_conditions = (u=u_bcs, v=no_slip, w=no_slip))

Ξ(z) = randn() * z / model.grid.Lz * (1 + z / model.grid.Lz) 
uᵢ(x, y, z) = sqrt(abs(Qᵘ)) * 1e-2 * Ξ(z)
Aᵢ(x, y, z) = ifelse(z==-1.0, 1.0, 0.0)
set!(model, u=uᵢ, w=uᵢ, v=uᵢ, A=Aᵢ)

simulation = Simulation(model, Δt=1.0, stop_time=1day)

wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=10minutes, diffusive_cfl=0.5)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|u|) = %.1e ms⁻¹, wall time: %s\n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                maximum(abs, sim.model.velocities.u), prettytime(sim.run_wall_time))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(20))

filepath = "empty"

simulation.output_writers[:profiles] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                     filename = "$filepath.jld2",
                     indices = (:, grid.Ny/2, :),
                     schedule = TimeInterval(1minute),
                     overwrite_existing = true)

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

hm_A = heatmap!(ax_A, xA, zA, Aₙ; colormap = :balance, colorrange = Alim)
Colorbar(fig[3, 4], hm_A; label = "m⁻³")

fig[1, 1:4] = Label(fig, title, textsize=24, tellwidth=false)

frames = intro:length(times)

record(fig, filepath * ".mp4", frames, framerate=16) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")
    n[] = i
end