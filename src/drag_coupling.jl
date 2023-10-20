using Oceananigans: set!, Center
using Oceananigans.Operators: volume

function update_tendencies!(bgc, particles::GiantKelp, model)
    Gᵘ, Gᵛ, Gʷ = particles.drag_fields

    set!(Gᵘ, 0)
    set!(Gᵛ, 0)
    set!(Gʷ, 0)

    parameters = particles.kinematic_parameters
    n_nodes = parameters.n_nodes

    #####
    ##### Calculate the drag from each individual
    ####
    # we have todo this first and serially for each particle otherwise we get a race condition adding to the drag field
    @inbounds @unroll for p in 1:length(particles)
        sf = particles.scalefactor[p]
        k_base = 1
        @unroll for n in 1:n_nodes
            i, j, k = particles.positions_ijk[p][n, :]
            vertical_spread = max(1, k - k_base  + 1)

            vol = volume(i, j, k, model.grid, Center(), Center(), Center()) * vertical_spread 
            total_scaling = sf / (vol * parameters.ρₒ)

            Gᵘ[i, j, k_base:k] = Gᵘ[i, j, k_base:k] .- particles.drag_forces[p][n, 1] * total_scaling
            Gᵛ[i, j, k_base:k] = Gᵛ[i, j, k_base:k] .- particles.drag_forces[p][n, 2] * total_scaling
            Gʷ[i, j, k_base:k] = Gʷ[i, j, k_base:k] .- particles.drag_forces[p][n, 3] * total_scaling

            k_base = k
        end
    end

    #####
    ##### Apply drag to model
    #####

    model.timestepper.Gⁿ.u .+= Gᵘ
    model.timestepper.Gⁿ.v .+= Gᵛ
    model.timestepper.Gⁿ.w .+= Gʷ
end