grid = RectilinearGrid(arch; size = (10, 10, 10), extent = (10, 10, 10))

holdfast_x = [5.]
holdfast_y = [5.]
holdfast_z = [-10.]

max_Δt = 1.0

number_nodes = 1
segment_unstretched_length = [10., ]

@inline function tracer_release(i, j, k, p, n, grid, clock, particles, tracers, parameters)
    C = tracers.C[i, j, k]
    
    return (parameters.base_value - C) / parameters.uptake_timescale / parameters.n_nodes
end

@inline analytical_concentration(t, scalefactor, parameters) = (1 - exp(- t * scalefactor / parameters.uptake_timescale / parameters.n_nodes / 10))

C = Forcing(tracer_release; parameters = (base_value = 1., uptake_timescale = 1hour, n_nodes = 2))

@testset "Tracer release" begin
    kelp = GiantKelp(; grid,
                    holdfast_x, holdfast_y, holdfast_z,
                    number_nodes,
                    segment_unstretched_length,
                    tracer_forcing = (; C))

    model = NonhydrostaticModel(; grid, 
                                tracers = (:C, ),
                                biogeochemistry = Biogeochemistry(NothingBGC(),
                                                                    particles = kelp),
                                advection = WENO())

    initial_positions = [0 0 10;]

    set!(kelp, positions = initial_positions)

    concentration_record = Float64[]

    Δt = 100.

    CUDA.@allowscalar for n in 1:500
        time_step!(model, Δt)
        push!(concentration_record, copy(model.tracers.C[6, 6, 10]))
    end


    @test all([isapprox(conc, analytical_concentration(n * Δt, 1, C.parameters), atol = 0.01) for (n, conc) in enumerate(concentration_record)])

    # check scale factors work
    kelp = GiantKelp(; grid,
                    holdfast_x, holdfast_y, holdfast_z,
                    number_nodes,
                    segment_unstretched_length,
                    scalefactor = [2., ],
                    tracer_forcing = (; C))

    model = NonhydrostaticModel(; grid, 
                                tracers = (:C, ),
                                biogeochemistry = Biogeochemistry(NothingBGC(),
                                                                    particles = kelp),
                                advection = WENO())

    initial_positions = [0 0 10;]

    set!(kelp, positions = initial_positions)

    concentration_record = Float64[]

    Δt = 100.

    CUDA.@allowscalar for n in 1:500
        time_step!(model, Δt)
        push!(concentration_record, copy(model.tracers.C[6, 6, 10]))
    end

    @test all([isapprox(conc, analytical_concentration(n * Δt, 2, C.parameters), atol = 0.01) for (n, conc) in enumerate(concentration_record)])
end