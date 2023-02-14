
@inline tension(Δx, l₀, Aᶜ, params) = Δx > l₀ && !(Δx == 0.0)  ? params.k * ((Δx - l₀) / l₀) ^ params.α * Aᶜ : 0.0


# This is only valid on a regularly spaced grid
# Benchmarks a lot lot faster than mean or sum()/dk etc. and about same speed as _interpolate which is weird
@inline function mean_squared_velocity(velocity::Field, i::Int, j::Int, k1::Int, k2::Int)
    res::Float64 = 0.0
    @unroll for k in k1:k2
        v = @inbounds velocity[i, j, k]
        res += v * abs(v)
    end
    return sign(res) * sqrt(abs(res)) / (k2 - k1 + 1)
end

@inline function mean_velocity(velocity::Field, i::Int, j::Int, k1::Int, k2::Int)
    res::Float64 = 0.0
    @unroll for k in k1:k2
        res += @inbounds velocity[i, j, k]
    end
    return res / (k2 - k1 + 1)
end

@kernel function step_node!(x_base, 
                            y_base, 
                            z_base, 
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

    if n == 1
        x⃗⁻ = zeros(3)
        u⃗ⁱ⁻¹ = zeros(3)
    else
        x⃗⁻ = @inbounds positions[p][n - 1, :]
        u⃗ⁱ⁻¹ = @inbounds velocities[p][n - 1, :]
    end

    Δx⃗ = x⃗ⁱ - x⃗⁻

    x, y, z = @inbounds [x_base[p], y_base[p], z_base[p]] + x⃗ⁱ

    l = sqrt(dot(Δx⃗, Δx⃗))
    Vᵖ = @inbounds pneumatocyst_volumes[p][n]

    Fᴮ = @inbounds 5 * Vᵖ * [0.0, 0.0, params.g] #currently assuming kelp is nutrally buoyant except for pneumatocysts

    if @inbounds Fᴮ[3] > 0 && z >= 0  # i.e. floating up not sinking, and outside of the surface
        @inbounds Fᴮ[3] = 0.0
    end

    Aᵇ = @inbounds blade_areas[p][n]
    rˢ = @inbounds stipe_radii[p][n]
    Vᵐ = π * rˢ ^ 2 * l + Aᵇ * 0.01 # TODO: change thickness to some realistic thing
    mᵉ = (Vᵐ + params.Cᵃ * (Vᵐ + Vᵖ)) * params.ρₒ + Vᵖ * (params.ρₒ - 500) 

    # we need ijk and this also reduces repetition of finding ijk
    i, j, k = fractional_indices(x, y, z, (Center(), Center(), Center()), water_velocities.u.grid)
    
    ξ, i = modf(i)
    η, j = modf(j)
    ζ, k = modf(k)

    i = Int(i + 1)
    j = Int(j + 1)
    k = Int(k + 1)

    @inbounds positions_ijk[p][n, :] = [i, j, k]

    ζ_base, k_base = @inbounds n == 1 ? (0.0, 1) : modf(1 +  fractional_z_index(positions[p][n - 1, 3] + z_base[p], Center(), water_velocities.u.grid)) # benchmarked and this is faster than ifelseing it

    k_base = Int(k_base)

    u⃗ʷ = @inbounds [ntuple(d -> mean_squared_velocity(water_velocities[d], i, j, k_base, k), 3)...] # [ntuple(d -> _interpolate(water_velocities[d], ξ, η, ζ, i, j, k), 3)...]
    u⃗ᵣₑₗ = u⃗ʷ - u⃗ⁱ
    sᵣₑₗ = sqrt(dot(u⃗ᵣₑₗ, u⃗ᵣₑₗ))

    a⃗ʷ = @inbounds [ntuple(d -> mean_velocity(water_accelerations[d], i, j, k_base, k), 3)...] #[ntuple(d -> mean_velocity(water_accelerations[d], i, j, k_base, k), 3)...]
    #a⃗ⁱ = @inbounds accelerations[p][n, :]
    #a⃗ᵣₑₗ = a⃗ʷ - a⃗ⁱ 

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

    #Δu⃗ⁱ⁻¹ = u⃗ⁱ⁻¹ - u⃗ⁱ
    #Δu⃗ⁱ⁺¹ = u⃗ⁱ⁺¹ - u⃗ⁱ

    l⁻ = sqrt(dot(Δx⃗⁻, Δx⃗⁻))
    l⁺ = sqrt(dot(Δx⃗⁺, Δx⃗⁺))

    T⁻ = tension(l⁻, l₀⁻, Aᶜ⁻, params) .* Δx⃗⁻ ./ (l⁻ + eps(0.0)) #+ ifelse(l⁻ > l₀⁻, params.kᵈ * Δu⃗ⁱ⁻¹, zeros(3))
    T⁺ = tension(l⁺, l₀⁺, Aᶜ⁺, params) .* Δx⃗⁺ ./ (l⁺ + eps(0.0)) #+ ifelse(l⁺ > l₀⁺, params.kᵈ * Δu⃗ⁱ⁺¹, zeros(3))

    Fⁱ = params. ρₒ * (Vᵐ + Vᵖ) .*  a⃗ʷ

    #if any(abs.(Fⁱ)/abs.(Fᴰ) .> 1000)
    #    @show Fⁱ
    #    @show Fᴰ
    #    Fⁱ *= maximum(abs, 1000 .* Fᴰ ./ Fⁱ)
    #end

    @inbounds begin 
        accelerations[p][n, :] .= (Fᴮ + Fᴰ + T⁻ + T⁺ + Fⁱ) ./ mᵉ - velocities[p][n, :] ./ params.τ
        drag_forces[p][n, :] .= Fᴰ + Fⁱ # store for back reaction onto water

        old_velocities[p][n, :] .= velocities[p][n, :]
        
        velocities[p][n, :] .+= timestepper(accelerations[p][n, :], old_accelerations[p][n, :], Δt, stage)
        
        old_accelerations[p][n, :] .= accelerations[p][n, :]

        positions[p][n, :] .+= timestepper(velocities[p][n, :], old_velocities[p][n, :], Δt, stage)

        if positions[p][n, 3] + z_base[p] > 0.0 #given above bouyancy conditions this should never be possible (assuming a flow with zero vertical velocity at the surface, i.e. a real one)
            positions[p][n, 3] = - z_base[p]
        end
    end
end

function kelp_dynamics!(particles, model, Δt)
    particles.properties.x .= particles.properties.fixed_x
    particles.properties.y .= particles.properties.fixed_y
    particles.properties.z .= particles.properties.fixed_z

    # calculate each particles node dynamics
    n_particles = length(particles)
    n_nodes = particles.parameters.n_nodes
    worksize = (n_particles, n_nodes)
    workgroup = (1, min(256, worksize[1]))

    step_kernel! = step_node!(device(model.architecture), workgroup, worksize)

    n_substeps = max(1, floor(Int, Δt / (particles.parameters.max_Δt)))

    water_accelerations = @inbounds model.timestepper.Gⁿ[(:u, :v, :w)]
    for substep in 1:n_substeps        
        for stage in stages(particles.parameters.timestepper)
            step_event = step_kernel!(particles.properties.x, 
                                      particles.properties.y, 
                                      particles.properties.z, 
                                      particles.properties.positions, 
                                      particles.properties.positions_ijk,
                                      particles.properties.velocities, 
                                      particles.properties.pneumatocyst_volumes, 
                                      particles.properties.stipe_radii,  
                                      particles.properties.blade_areas, 
                                      particles.properties.relaxed_lengths, 
                                      particles.properties.accelerations, 
                                      particles.properties.drag_forces, 
                                      particles.properties.old_velocities, 
                                      particles.properties.old_accelerations, 
                                      model.velocities,
                                      water_accelerations, 
                                      Δt / n_substeps, 
                                      particles.parameters,
                                      particles.parameters.timestepper,
                                      stage)

            wait(step_event)
        end
    end
    particles.parameters.other_dynamics(particles, model, Δt)
end