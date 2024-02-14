function apply_drag!(particles, Gᵘ, Gᵛ, Gʷ, i, j, k_top, k_base, total_mass, p, n)
    @inbounds begin
        sf = particles.scalefactor[p]

        total_scaling = sf / total_mass

        for k in k_base:k_top
            Atomix.@atomic Gᵘ[i, j, k] = Gᵘ[i, j, k] - particles.drag_forces[p, n, 1] * total_scaling
            Atomix.@atomic Gᵛ[i, j, k] = Gᵛ[i, j, k] - particles.drag_forces[p, n, 2] * total_scaling
            Atomix.@atomic Gʷ[i, j, k] = Gʷ[i, j, k] - particles.drag_forces[p, n, 3] * total_scaling
            #atomic_add!(Gᵘ, i, j, k, - particles.drag_forces[p, n, 1] * total_scaling)
            #atomic_add!(Gᵛ, i, j, k, - particles.drag_forces[p, n, 2] * total_scaling)
            #atomic_add!(Gʷ, i, j, k, - particles.drag_forces[p, n, 3] * total_scaling)
        end
    end
end
