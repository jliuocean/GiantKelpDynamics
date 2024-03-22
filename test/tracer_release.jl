holdfast_x = [5.]
holdfast_y = [5.]
holdfast_z = [-10.]

max_Δt = 1.0

number_nodes = 1
segment_unstretched_length = [10., ]

@inline constant_tracer_release(i, j, k, p, n, grid, clock, particles, tracers, parameters) = 0.1 / Oceananigans.Operators.volume(i, j, k, grid, Center(), Center(), Center())

@inline scaled_tracer_release(i, j, k, p, n, grid, clock, particles, tracers, parameters) = parameters * 0.1 / Oceananigans.Operators.volume(i, j, k, grid, Center(), Center(), Center())

@inline total_released_tracer(t) = 0.1 * t

@testset "Tracer release" begin
    for horizontal_resolution in (10, 20), vertical_resolution in (10, 20)
        grid = RectilinearGrid(arch; size = (horizontal_resolution, horizontal_resolution, vertical_resolution), extent = (10, 10, 10))

        kelp = GiantKelp(; grid,
                        holdfast_x, holdfast_y, holdfast_z,
                        number_nodes,
                        segment_unstretched_length,
                        tracer_forcing = (; C = Forcing(constant_tracer_release)))

        model = NonhydrostaticModel(; grid, 
                                    tracers = (:C, ),
                                    biogeochemistry = Biogeochemistry(NothingBGC(),
                                                                        particles = kelp),
                                    advection = WENO())

        initial_positions = [0 0 10;]

        set!(kelp, positions = initial_positions)

        concentration_record = Float64[]

        Δt = 100.

        Oceananigans.TimeSteppers.update_state!(model)
        Oceananigans.Models.LagrangianParticleTracking.update_lagrangian_particle_properties!(kelp, model, model.biogeochemistry, Δt)

        CUDA.@allowscalar for n in 1:500
            time_step!(model, Δt)
            push!(concentration_record, sum([model.tracers.C[kelp.positions_ijk[1, 1, 1], kelp.positions_ijk[1, 1, 2], k] * Oceananigans.Operators.volume(1, 1, k, grid, Center(), Center(), Center()) for k=1:grid.Nz]))
        end

        @test all([isapprox(conc, total_released_tracer(n * Δt), atol = 0.01) for (n, conc) in enumerate(concentration_record)])
    end

    # check scale factor and parametrers work
    grid = RectilinearGrid(arch; size = (10, 10, 10), extent = (10, 10, 10))

    kelp = GiantKelp(; grid,
                      holdfast_x, holdfast_y, holdfast_z,
                      number_nodes,
                      segment_unstretched_length,
                      scalefactor = [2., ],
                      tracer_forcing = (; C = Forcing(scaled_tracer_release; parameters = 0.5)))

    model = NonhydrostaticModel(; grid, 
                                tracers = (:C, ),
                                biogeochemistry = Biogeochemistry(NothingBGC(),
                                                                    particles = kelp),
                                advection = WENO())

    initial_positions = [0 0 10;]

    set!(kelp, positions = initial_positions)

    concentration_record = Float64[]

    Δt = 50.

    Oceananigans.TimeSteppers.update_state!(model)
    Oceananigans.Models.LagrangianParticleTracking.update_lagrangian_particle_properties!(kelp, model, model.biogeochemistry, Δt)

    CUDA.@allowscalar for n in 1:500
        time_step!(model, Δt)
        push!(concentration_record, sum([model.tracers.C[kelp.positions_ijk[1, 1, 1], kelp.positions_ijk[1, 1, 2], k] * Oceananigans.Operators.volume(1, 1, k, grid, Center(), Center(), Center()) for k=1:grid.Nz]))
    end

    @test all([isapprox(conc, total_released_tracer(n * Δt), atol = 0.01) for (n, conc) in enumerate(concentration_record)])
end