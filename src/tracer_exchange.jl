struct TracerExchange{P, FT}
    particles :: P
    uptake_timescale :: FT
    base_value :: FT
end

function (exchange::TracerExchange)(model)
    O = model.tracers.O

    properties = exchange.particles.properties
    parameters = exchange.particles.parameters


    @inbounds @unroll for p in 1:length(model.particles)
        sf = properties.scalefactor[p]

        k_base = 1
        @unroll for n in 1:parameters.n_nodes
            i, j, k = properties.positions_ijk[p][n, :]
            vertical_spread = max(1, k - k_base  + 1)

            total_scaling = sf / vertical_spread

            for kidx in k_base:k
                model.timestepper.Gⁿ.O[i, j, kidx] -= (total_scaling / exchange.uptake_timescale) .* (O[i, j, kidx] .- exchange.base_value)
            end

            k_base = k
        end
    end
end