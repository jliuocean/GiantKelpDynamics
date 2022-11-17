using Oceananigans, StructArrays
using Oceananigans.Units

include("macrosystis_dynamics.jl")

nodes = Nodes([0.0 0.0 4.0; ], zeros(1, 3), [4.0], [0.03], zeros(Int64, 1), zeros(1), [0.002], [1.0], zeros(1, 3), zeros(1, 3), zeros(1, 3), zeros(1, 3))
particle_struct = StructArray{GiantKelp}(([4.0], [0.0], [-8.0], [4.0], [4.0], [-8.0], [nodes]))

function guassian_smoothing(r, z, rᵉ)
    if z>0
        r = sqrt(r^2 + z^2)
    end

    return exp(-(7*r)^2/(2*rᵉ^2))/sqrt(2*π*rᵉ^2)
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

Lx, Ly, Lz = 8, 8, 8
Nx, Ny, Nz = 8 .*(Lx, Ly, Lz)
grid = RectilinearGrid(size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz), topology=(Periodic, Periodic, Bounded))

drag_nodes = repeat([CenterField(grid)], 1, 1)
drag_normalisation = repeat([Inf], 1, 1)

model = NonhydrostaticModel(;grid, particles, auxiliary_fields = (; drag_nodes, drag_normalisation))

set!(model, u=0.15)

drag_water_callback = Callback(drag_water!; callsite = TendencyCallsite())

simulation = Simulation(model, Δt=0.05, stop_time=1minutes)

simulation.callbacks[:drag_water] = drag_water_callback

run!(simulation)