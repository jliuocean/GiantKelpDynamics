struct DiscreteDrag{FT}
    field :: FT

    function DiscreteDrag(; grid)
        field = CenterField(grid)
        FT = typeof(field)

        return new{FT}(field)
    end
end

struct DiscreteDragSet{DU, DV, DW, SD}
    u :: DU
    v :: DV
    w :: DW

    xy_smudge_distance :: SD # not using at the moment but might add back in

    function DiscreteDragSet(; grid, xy_smudge_distance::SD = 0) where {SD}
        drag_tuple = ntuple(n -> DiscreteDrag(; grid), 3)   
        DU, DV, DW = typeof.(drag_tuple)
        return new{DU, DV, DW, SD}(drag_tuple..., xy_smudge_distance)
    end
end


@inline (drag::DiscreteDrag)(i, j, k, grid, clock, model_fields) = @inbounds drag.field[i, j, k]

@inline function (drag_set::DiscreteDragSet)(model)
    drag_set.u.field .= 0.0
    drag_set.v.field .= 0.0
    drag_set.w.field .= 0.0

    properties = model.particles.properties
    parameters = model.particles.parameters

    @inbounds @unroll for p in 1:length(model.particles)
        sf = properties.scalefactor[p]
        k_base = 1
        @unroll for n in 1:parameters.n_nodes
            i, j, k = properties.positions_ijk[p][n, :]
            vertical_spread = max(1, k - k_base  + 1)

            vol = Vᶜᶜᶜ(i, j, k, model.grid) * vertical_spread #sum(ntuple(n -> Vᶜᶜᶜ(i, j, k + n - 1, model.grid), vertical_spread))
            total_scaling = sf / (vol * parameters.ρₒ)

            drag_set.u.field[i, j, k_base:k] = drag_set.u.field[i, j, k_base:k] .- properties.drag_forces[p][n, 1] * total_scaling
            drag_set.v.field[i, j, k_base:k] = drag_set.v.field[i, j, k_base:k] .- properties.drag_forces[p][n, 2] * total_scaling
            drag_set.w.field[i, j, k_base:k] = drag_set.w.field[i, j, k_base:k] .- properties.drag_forces[p][n, 3] * total_scaling

            k_base = k
        end
    end
end