function apply_drag!(particles, Gᵘ, Gᵛ, Gʷ, i, j, k, k_base, volume, p, n)
    @inbounds begin
        sf = particles.scalefactor[p]
        parameters = particles.kinematic_parameters

        total_scaling = sf / (volume * parameters.ρₒ)

        Gᵘ[i, j, k_base:k] = Gᵘ[i, j, k_base:k] .- particles.drag_forces[p][n, 1] * total_scaling
        Gᵛ[i, j, k_base:k] = Gᵛ[i, j, k_base:k] .- particles.drag_forces[p][n, 2] * total_scaling
        Gʷ[i, j, k_base:k] = Gʷ[i, j, k_base:k] .- particles.drag_forces[p][n, 3] * total_scaling
    end
end