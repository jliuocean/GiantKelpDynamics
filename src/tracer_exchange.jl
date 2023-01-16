struct TracerExchange{P, FT}
    particles :: P
    uptake_half_saturation :: FT
    excretion_rate :: FT
end

function (exchange::TracerExchange)(model)
    U = model.tracers.U
    O = model.tracers.O

    properties = exchange.particles.properties
    parameters = exchange.particles.parameters


    @inbounds @unroll for p in 1:length(model.particles)
        sf = properties.scalefactor[p]

        k_base = 1
        @unroll for n in 1:parameters.n_nodes
            i, j, k = properties.positions_ijk[p][n, :]
            #vertical_spread = max(1, k - k_base  + 1)

            total_scaling = 0.0001 * sf #/ vertical_spread

            U[i, j, k_base:k] = U[i, j, k_base:k] - total_scaling .* U[i, j, k_base:k] ./ (U[i, j, k_base:k] .+ exchange.uptake_half_saturation)
            O[i, j, k_base:k] = O[i, j, k_base:k] .+ total_scaling * exchange.excretion_rate

            k_base = k
        end
    end
end