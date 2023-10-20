grid = RectilinearGrid(size = (128, 128, 8), extent = (500, 500, 8))

spacing = 100.

x_pattern = 200.:spacing:300.
y_pattern = 200.:spacing:300.

holdfast_x = vec([x for x in x_pattern, y in y_pattern])
holdfast_y = vec([y for x in x_pattern, y in y_pattern])
holdfast_z = vec([-8. for x in x_pattern, y in y_pattern])

number_nodes = 2
segment_unstretched_length = [16, 8]

kelp = GiantKelp(; grid,
                holdfast_x, holdfast_y, holdfast_z,
                number_nodes,
                segment_unstretched_length)

model = NonhydrostaticModel(; grid, 
                            biogeochemistry = Biogeochemistry(NothingBGC(),
                                                                particles = kelp),
                            advection = WENO())

@testset "Set" begin
    initial_positions = [0 0 8; 8 0 8]

    set!(kelp, positions = initial_positions)

    @test all([all(position .== initial_positions) for position in kelp.positions])

    initial_positions = vec([[i 0 8; 8 0 8] for (i, x) in enumerate(x_pattern), (j, y) in enumerate(y_pattern)])

    set!(kelp, positions = initial_positions)

    @test all(kelp.positions .== initial_positions)
end

@testset "Output" begin
    simulation = Simulation(model, Δt = 1, stop_iteration = model.clock.iteration + 10)

    simulation.output_writers[:kelp] = JLD2OutputWriter(model, (; positions = kelp.positions, blade_areas = kelp.blade_areas), overwrite_existing = true, schedule = IterationInterval(1), filename = "kelp.jld2")
    
    run!(simulation)

    file = jldopen("kelp.jld2")

    @test keys(file["timeseries"]) == ["positions", "blade_areas", "t"]

    indices = keys(file["timeseries/t"])

    positions = [file["timeseries/positions/$idx"] for idx in indices]
    blade_areas = [file["timeseries/blade_areas/$idx"] for idx in indices]

    close(file)

    @test length(positions[1]) == length(kelp)

    @test all(positions[end][1] .== kelp.positions[1])

    @test all(blade_areas[end][1] .== kelp.blade_areas[1])
end