grid = RectilinearGrid(size = (128, 128, 8), extent = (500, 500, 8))

spacing = 100.

x_pattern = 200.:spacing:300.
y_pattern = 200.:spacing:300.

holdfast_x = vec([x for x in x_pattern, y in y_pattern])
holdfast_y = vec([y for x in x_pattern, y in y_pattern])
holdfast_z = vec([-8. for x in x_pattern, y in y_pattern])

number_nodes = 2
segment_unstretched_length = [16, 8]

@testset "Kelp move" begin
    kelp = GiantKelp(; grid,
                    holdfast_x, holdfast_y, holdfast_z,
                    number_nodes,
                    segment_unstretched_length)

    model = NonhydrostaticModel(; grid, 
                                biogeochemistry = Biogeochemistry(NothingBGC(),
                                                                    particles = kelp),
                                advection = WENO())


    initial_positions = [0 0 8; 8 0 8]

    set!(kelp, positions = initial_positions)

    time_step!(model, 10.)

    # not moving when no flow and unstretched
    @test all([all(position .== initial_positions) for position in kelp.positions])

    initial_positions = [15 0 8; 25 0 8]

    set!(kelp, positions = initial_positions)

    position_record = []
    for n in 1:100
        push!(position_record, copy(kelp.positions[1]))
        time_step!(model, 1.)
    end

    # nodes are setteling
    @test all(isapprox.(position_record[end - 1], position_record[end]; atol = 0.001))
end

@testset "Drag" begin
    scalefactor = ones(length(holdfast_x))

    kelp = GiantKelp(; grid,
                       holdfast_x, holdfast_y, holdfast_z,
                       number_nodes,
                       segment_unstretched_length,
                       scalefactor)

    model = NonhydrostaticModel(; grid, 
                                biogeochemistry = Biogeochemistry(NothingBGC(),
                                                                    particles = kelp),
                                advection = WENO())

    initial_positions = [0 0 8; 8 0 8]
    u₀ = 0.2

    set!(kelp, positions = initial_positions)
    set!(model, u = u₀)

    for n in 1:1000
        time_step!(model, 1.)
    end

    # the kelp are being moved by the flow
    @test all([all(position[:, 1:2] .!= initial_positions[:, 1:2]) for position in kelp.positions])

    # the kelp are dragging the water
    @test !(mean(model.velocities.u) ≈ u₀)
    @test maximum(abs, model.velocities.v) .!= 0
end