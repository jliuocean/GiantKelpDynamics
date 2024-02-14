"""
    UtterDenny(; spring_constant = 1.91 * 10 ^ 7,
                 spring_exponent = 1.41,
                 water_density = 1026.0,
                 pneumatocyst_specific_buoyancy = 5.,
                 gravitational_acceleration = 9.81,
                 stipe_drag_coefficient = 1.,
                 blade_drag_coefficient = 0.4 * 12 ^ -0.485,
                 added_mass_coefficient = 3.,
                 damping_timescale = 5.)

Sets up the kinematic model for giant kelp motion from [Utter1996](@citet) and [Rosman2013](@citet).
"""

@kwdef struct UtterDenny{FT}
                 spring_constant :: FT = 1.91 * 10 ^ 7
                 spring_exponent :: FT = 1.41
                   water_density :: FT = 1026.0
  pneumatocyst_specific_buoyancy :: FT = 5.
      gravitational_acceleration :: FT = 9.81
          stipe_drag_coefficient :: FT = 1.
          blade_drag_coefficient :: FT = 0.87
          added_mass_coefficient :: FT = 3.
               damping_timescale :: FT = 5.
end

@kernel function (kinematics::UtterDenny)(x_holdfast, y_holdfast, z_holdfast, 
                                          positions, positions_ijk,
                                          velocities, 
                                          pneumatocyst_volumes, stipe_radii, 
                                          blade_areas, relaxed_lengths, 
                                          accelerations, drag_forces, 
                                          water_velocities, water_accelerations,
                                          kinematics)
    p, n = @index(Global, NTuple)

    spring_constant = kinematics.spring_constant
    spring_exponent = kinematics.spring_exponent
    ρₒ = kinematics.water_density
    ρₚ = kinematics.pneumatocyst_specific_buoyancy
    g = kinematics.gravitational_acceleration
    Cᵈˢ = kinematics.stipe_drag_coefficient
    Cᵈᵇ = kinematics.blade_drag_coefficient
    Cᵃ = kinematics.added_mass_coefficient
    τ = kinematics.damping_timescale

    x⃗ⁱ = @inbounds positions[p, n, :]
    u⃗ⁱ = @inbounds velocities[p, n, :]

    # can we eliminate this branching logic
    if n == 1
        x⃗⁻ = zeros(3)
        u⃗ⁱ⁻¹ = zeros(3)
    else
        x⃗⁻ = @inbounds positions[p, n - 1, :]
        u⃗ⁱ⁻¹ = @inbounds velocities[p, n - 1, :]
    end

    Δx⃗ = x⃗ⁱ - x⃗⁻

    x, y, z = @inbounds [x_holdfast[p], y_holdfast[p], z_holdfast[p]] + x⃗ⁱ

    l = sqrt(dot(Δx⃗, Δx⃗))
    Vᵖ = @inbounds pneumatocyst_volumes[p, n]

    Fᴮ = @inbounds ρₚ * Vᵖ * [0.0, 0.0, g]

    if @inbounds Fᴮ[3] > 0 && z >= 0  # i.e. floating up not sinking, and outside of the surface
        @inbounds Fᴮ[3] = 0.0
    end

    Aᵇ = @inbounds blade_areas[p, n]
    rˢ = @inbounds stipe_radii[p, n]
    Vᵐ = π * rˢ ^ 2 * l + Aᵇ * 0.01
    mᵉ = (Vᵐ + Cᵃ * (Vᵐ + Vᵖ)) * ρₒ + Vᵖ * (ρₒ - 500) 

    # we need ijk and this also reduces repetition of finding ijk
    i, j, k = fractional_indices(x, y, z, (Center(), Center(), Center()), water_velocities.u.grid)

    _, i = modf(i)
    _, j = modf(j)
    _, k = modf(k)

    i = Int(i + 1)
    j = Int(j + 1)
    k = Int(k + 1)

    @inbounds positions_ijk[p, n, :] = [i, j, k]

    _, k1 = @inbounds ifelse(n == 1, (0.0, 1), modf(1 +  fractional_z_index(positions[p, n - 1, 3] + z_holdfast[p], (Center(), Center(), Center()), water_velocities.u.grid)))

    k1 = Int(k1)

    u⃗ʷ = @inbounds [mean_squared_field(water_velocities[1], i, j, k1, k),
                    mean_squared_field(water_velocities[2], i, j, k1, k),
                    mean_squared_field(water_velocities[3], i, j, k1, k)]
    u⃗ᵣₑₗ = u⃗ʷ - u⃗ⁱ
    sᵣₑₗ = sqrt(dot(u⃗ᵣₑₗ, u⃗ᵣₑₗ))

    a⃗ʷ = @inbounds [mean_squared_field(water_accelerations[1], i, j, k1, k),
                    mean_squared_field(water_accelerations[2], i, j, k1, k),
                    mean_squared_field(water_accelerations[3], i, j, k1, k)]

    θ = acos(min(1, abs(dot(u⃗ᵣₑₗ, Δx⃗)) / (sᵣₑₗ * l + eps(0.0))))
    Aˢ = @inbounds 2 * rˢ * l * abs(sin(θ)) + π * rˢ * abs(cos(θ))

    Fᴰ = 0.5 * ρₒ * (Cᵈˢ * Aˢ + Cᵈᵇ * Aᵇ) * sᵣₑₗ * u⃗ᵣₑₗ

    if n == @inbounds size(relaxed_lengths, 2)
        x⃗⁺ = x⃗ⁱ 
        u⃗ⁱ⁺¹ = u⃗ⁱ
        Aᶜ⁺ = 0.0 
        l₀⁺ = @inbounds relaxed_lengths[p, n] 
    else
        x⃗⁺ = @inbounds positions[p, n + 1, :]
        u⃗ⁱ⁺¹ = @inbounds velocities[p, n + 1, :]
        Aᶜ⁺ = @inbounds π * stipe_radii[p, n + 1] ^ 2
        l₀⁺ = @inbounds relaxed_lengths[p, n + 1]
    end

    Aᶜ⁻ = @inbounds π * rˢ ^ 2
    l₀⁻ = @inbounds relaxed_lengths[p, n]

    Δx⃗⁻ = x⃗⁻ - x⃗ⁱ
    Δx⃗⁺ = x⃗⁺ - x⃗ⁱ

    l⁻ = sqrt(dot(Δx⃗⁻, Δx⃗⁻))
    l⁺ = sqrt(dot(Δx⃗⁺, Δx⃗⁺))

    T⁻ = tension(l⁻, l₀⁻, Aᶜ⁻, spring_constant, spring_exponent) * Δx⃗⁻ / (l⁻ + eps(0.0))
    T⁺ = tension(l⁺, l₀⁺, Aᶜ⁺, spring_constant, spring_exponent) * Δx⃗⁺ / (l⁺ + eps(0.0))

    Fⁱ =  ρₒ * (Vᵐ + Vᵖ) *  a⃗ʷ

    total_acceleraiton = (Fᴮ + Fᴰ + T⁻ + T⁺ + Fⁱ) / mᵉ

    @inbounds for d=1:3 
        accelerations[p, n, d] = total_acceleraiton[d] - velocities[p, n, d] / τ
        drag_forces[p, n, d] = Fᴰ[d] # store for back reaction onto water
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

@inline tension(Δx, l₀, Aᶜ, k, α) = ifelse(Δx > l₀ && !(Δx == 0.0), k * ((Δx - l₀) / l₀) ^ α * Aᶜ, 0.0)
