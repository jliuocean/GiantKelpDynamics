using LinearAlgebra

using Oceananigans.Fields: fractional_indices, fractional_z_index, _interpolate

function update_lagrangian_particle_properties!(particles::GiantKelp, model, bgc, Δt)
    # this will need to be modified when we have biological properties to update
    n_particles = length(particles)
    parameters = particles.kinematic_parameters
    n_nodes = parameters.n_nodes
    worksize = (n_particles, n_nodes)
    workgroup = (1, min(256, worksize[1]))

    step_kernel! = step_node!(device(model.architecture), workgroup, worksize)

    n_substeps = max(1, 1 + floor(Int, Δt / (particles.max_Δt)))

    water_accelerations = @inbounds model.timestepper.Gⁿ[(:u, :v, :w)]
    for _ in 1:n_substeps, stage in stages(particles.timestepper)
        step_kernel!(particles.holdfast_x, 
                     particles.holdfast_y, 
                     particles.holdfast_z, 
                     particles.positions, 
                     particles.positions_ijk,
                     particles.velocities, 
                     particles.pneumatocyst_volumes, 
                     particles.stipe_radii,  
                     particles.blade_areas, 
                     particles.relaxed_lengths, 
                     particles.accelerations, 
                     particles.drag_forces, 
                     particles.old_velocities, 
                     particles.old_accelerations, 
                     model.velocities,
                     water_accelerations, 
                     Δt / n_substeps, 
                     parameters,
                     particles.timestepper,
                     stage)
    end

    synchronize(device(architecture(model)))

    particles.custom_dynamics(particles, model, Δt)
end


@kernel function step_node!(x_holdfast, 
                            y_holdfast, 
                            z_holdfast, 
                            positions, 
                            positions_ijk,
                            velocities, 
                            pneumatocyst_volumes, 
                            stipe_radii, 
                            blade_areas,
                            relaxed_lengths, 
                            accelerations, 
                            drag_forces, 
                            old_velocities,
                            old_accelerations, 
                            water_velocities, 
                            water_accelerations, 
                            Δt, 
                            params,
                            timestepper, 
                            stage)

    p, n = @index(Global, NTuple)

    x⃗ⁱ = @inbounds positions[p][n, :]
    u⃗ⁱ = @inbounds velocities[p][n, :]

    # can we eliminate this branching logic
    if n == 1
        x⃗⁻ = zeros(3)
        u⃗ⁱ⁻¹ = zeros(3)
    else
        x⃗⁻ = @inbounds positions[p][n - 1, :]
        u⃗ⁱ⁻¹ = @inbounds velocities[p][n - 1, :]
    end

    Δx⃗ = x⃗ⁱ - x⃗⁻

    x, y, z = @inbounds [x_holdfast[p], y_holdfast[p], z_holdfast[p]] + x⃗ⁱ

    l = sqrt(dot(Δx⃗, Δx⃗))
    Vᵖ = @inbounds pneumatocyst_volumes[p][n]

    Fᴮ = @inbounds 5 * Vᵖ * [0.0, 0.0, params.g] #currently assuming kelp is nutrally buoyant except for pneumatocysts

    if @inbounds Fᴮ[3] > 0 && z >= 0  # i.e. floating up not sinking, and outside of the surface
        @inbounds Fᴮ[3] = 0.0
    end

    Aᵇ = @inbounds blade_areas[p][n]
    rˢ = @inbounds stipe_radii[p][n]
    Vᵐ = π * rˢ ^ 2 * l + Aᵇ * 0.01
    mᵉ = (Vᵐ + params.Cᵃ * (Vᵐ + Vᵖ)) * params.ρₒ + Vᵖ * (params.ρₒ - 500) 

    # we need ijk and this also reduces repetition of finding ijk
    i, j, k = fractional_indices(x, y, z, (Center(), Center(), Center()), water_velocities.u.grid)
    
    _, i = modf(i)
    _, j = modf(j)
    _, k = modf(k)

    i = Int(i + 1)
    j = Int(j + 1)
    k = Int(k + 1)

    @inbounds positions_ijk[p][n, :] = [i, j, k]

    _, k1 = @inbounds n == 1 ? (0.0, 1) : modf(1 +  fractional_z_index(positions[p][n - 1, 3] + z_holdfast[p], (Center(), Center(), Center()), water_velocities.u.grid))

    k1 = Int(k1)

    u⃗ʷ = @inbounds [ntuple(d -> mean_squared_field(water_velocities[d], i, j, k1, k), 3)...]
    u⃗ᵣₑₗ = u⃗ʷ - u⃗ⁱ
    sᵣₑₗ = sqrt(dot(u⃗ᵣₑₗ, u⃗ᵣₑₗ))

    a⃗ʷ = @inbounds [ntuple(d -> mean_field(water_accelerations[d], i, j, k1, k), 3)...]

    θ = acos(min(1, abs(dot(u⃗ᵣₑₗ, Δx⃗)) / (sᵣₑₗ * l + eps(0.0))))
    Aˢ = @inbounds 2 * rˢ * l * abs(sin(θ)) + π * rˢ * abs(cos(θ))

    Fᴰ = 0.5 * params.ρₒ * (params.Cᵈˢ * Aˢ + params.Cᵈᵇ * Aᵇ) * sᵣₑₗ .* u⃗ᵣₑₗ

    if n == @inbounds length(relaxed_lengths[p])
        x⃗⁺ = x⃗ⁱ 
        u⃗ⁱ⁺¹ = u⃗ⁱ
        Aᶜ⁺ = 0.0 
        l₀⁺ = @inbounds relaxed_lengths[p][n] 
    else
        x⃗⁺ = @inbounds positions[p][n + 1, :]
        u⃗ⁱ⁺¹ = @inbounds velocities[p][n + 1, :]
        Aᶜ⁺ = @inbounds π * stipe_radii[p][n + 1] ^ 2
        l₀⁺ = @inbounds relaxed_lengths[p][n + 1]
    end

    Aᶜ⁻ = @inbounds π * rˢ ^ 2
    l₀⁻ = @inbounds relaxed_lengths[p][n]

    Δx⃗⁻ = x⃗⁻ - x⃗ⁱ
    Δx⃗⁺ = x⃗⁺ - x⃗ⁱ

    l⁻ = sqrt(dot(Δx⃗⁻, Δx⃗⁻))
    l⁺ = sqrt(dot(Δx⃗⁺, Δx⃗⁺))

    T⁻ = tension(l⁻, l₀⁻, Aᶜ⁻, params) .* Δx⃗⁻ ./ (l⁻ + eps(0.0))
    T⁺ = tension(l⁺, l₀⁺, Aᶜ⁺, params) .* Δx⃗⁺ ./ (l⁺ + eps(0.0))

    Fⁱ = params. ρₒ * (Vᵐ + Vᵖ) .*  a⃗ʷ

    @inbounds begin 
        accelerations[p][n, :] .= (Fᴮ + Fᴰ + T⁻ + T⁺ + Fⁱ) ./ mᵉ - velocities[p][n, :] ./ params.τ
        drag_forces[p][n, :] .= Fᴰ + Fⁱ # store for back reaction onto water

        old_velocities[p][n, :] .= velocities[p][n, :]
        
        velocities[p][n, :] .+= timestepper(accelerations[p][n, :], old_accelerations[p][n, :], Δt, stage)
        
        old_accelerations[p][n, :] .= accelerations[p][n, :]

        positions[p][n, :] .+= timestepper(velocities[p][n, :], old_velocities[p][n, :], Δt, stage)

        if positions[p][n, 3] + z_holdfast[p] > 0.0 #given above bouyancy conditions this should never be possible (assuming a flow with zero vertical velocity at the surface, i.e. a real one)
            positions[p][n, 3] = - z_holdfast[p]
        end
    end
end

# This is only valid on a regularly spaced grid
# Benchmarks a lot lot faster than mean or sum()/dk etc. and about same speed as _interpolate which is weird
@inline function mean_squared_field(velocity::Field, i::Int, j::Int, k1::Int, k2::Int)
    res = 0.0
    @unroll for k in k1:k2
        v = @inbounds velocity[i, j, k]
        res += v * abs(v)
    end
    return sign(res) * sqrt(abs(res)) / (k2 - k1 + 1)
end

@inline function mean_field(velocity::Field, i::Int, j::Int, k1::Int, k2::Int)
    res = 0.0
    @unroll for k in k1:k2
        res += @inbounds velocity[i, j, k]
    end
    return res / (k2 - k1 + 1)
end

@inline tension(Δx, l₀, Aᶜ, params) = Δx > l₀ && !(Δx == 0.0)  ? params.k * ((Δx - l₀) / l₀) ^ params.α * Aᶜ : 0.0
