grid = RectilinearGrid(arch; size = (128, 128, 8), extent = (500, 500, 8))

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

    @test all([all(Array(kelp.positions[p, :, :]) .== initial_positions) for p in 1:length(kelp)])

    initial_positions = Array(similar(kelp.positions))

    for p in 1:length(kelp)
        initial_positions[p, :, :] .= [p 0 8; 8 0 8]
    end

    set!(kelp, positions = initial_positions)

    @test all(Array(kelp.positions) .== initial_positions)
end

@testset "Output" begin
    simulation = Simulation(model, Δt = 1, stop_iteration = model.clock.iteration + 10)

    simulation.output_writers[:kelp] = JLD2OutputWriter(model, (; positions = kelp.positions, blade_areas = kelp.blade_areas), overwrite_existing = true, schedule = IterationInterval(1), filename = "kelp.jld2")
    
    run!(simulation)

    # TODO: make some utility to load this stuff

    file = jldopen("kelp.jld2")

    @test keys(file["timeseries"]) == ["positions", "blade_areas", "t"]

    indices = keys(file["timeseries/t"])

    positions = [file["timeseries/positions/$idx"] for idx in indices]
    blade_areas = [file["timeseries/blade_areas/$idx"] for idx in indices]

    close(file)

    @test all(positions[end][1, :, :] .== Array(kelp.positions[1, :, :]))

    @test all(blade_areas[end][1, :, :] .== Array(kelp.blade_areas[1, :, :]))
end