grid = RectilinearGrid(arch; size = (128, 128, 8), extent = (500, 500, 8))

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
    CUDA.@allowscalar @test all([all(Array(kelp.positions[p, :, :]) .== initial_positions) for p=1:length(holdfast_x)])

    initial_positions = [15 0 8; 25 0 8]

    set!(kelp, positions = initial_positions)

    for n in 1:200
        time_step!(model, 1.)
    end

    position_record = copy(kelp.positions)

    # nodes are setteling
    CUDA.@allowscalar @test all(isapprox.(position_record, Array(kelp.positions); atol = 0.001))
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

    all_initial_positions = copy(kelp.positions)

    for n in 1:200
        time_step!(model, 1.)
    end

    # the kelp are being moved by the flow
    
    CUDA.@allowscalar  @test !any(isapprox.(all_initial_positions[:, :, 1:2], Array(kelp.positions[:, :, 1:2]); atol = 0.001))

    # the kelp are dragging the water
    @test !(mean(model.velocities.u) ≈ u₀)
    @test !isapprox(maximum(abs, model.velocities.v), 0)
end