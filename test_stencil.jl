using Oceananigans, StructArrays
using Oceananigans.Units

include("macrosystis_dynamics.jl")

nodes = Nodes([0.0 0.0 2.0; 1.0 0.0 4.0; 3.0 0.0 4.0], zeros(3, 3), [2.0; 2.0; 2.0], [0.03; 0.03; 0.03], zeros(Int64, 3), zeros(3), [0.002; 0.002; 0.002], [0.5, 0.5, 0.5], zeros(3, 3), zeros(3, 3), zeros(3, 3), zeros(3, 3))
particle_struct = StructArray{GiantKelp}(([4.0], [4.0], [-8.0], [4.0], [4.0], [-8.0], [nodes]))

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
                                          n_nodes = 3))

Lx, Ly, Lz = 8, 8, 8
Nx, Ny, Nz = 8 .*(Lx, Ly, Lz)
grid = RectilinearGrid(size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz), topology=(Periodic, Periodic, Bounded))

drag_nodes = CenterField(grid)

model = NonhydrostaticModel(;grid, particles, auxiliary_fields = (; drag_nodes))

set!(model, u=0.15)
drag_water_callback = Callback(drag_water!; callsite = TendencyCallsite())
#=


simulation = Simulation(model, Δt=0.05, stop_time=1minutes)

simulation.callbacks[:drag_water] = drag_water_callback

run!(simulation)=#