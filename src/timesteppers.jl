"""
    RK3(; γ :: G = (8//15, 5//12, 3//4),
          ζ :: Z = (0.0, -17//60, -5//12)

Holds parameters for a third-order Runge-Kutta-Wray time-stepping scheme described by Le and Moin (1991).
"""
struct RK3{G, Z}
    γ :: G
    ζ :: Z

    function RK3(; γ :: G = (8//15, 5//12, 3//4),
                   ζ :: Z = (0.0, -17//60, -5//12)) where {G, Z}
        return new{G, Z}(γ, ζ)
    end
end

@inline function (ts::RK3)(u⃗, u⃗⁻, Δt, stage)
    return @inbounds Δt * ts.γ[stage] * u⃗ + Δt * ts.ζ[stage] * u⃗⁻
end

"""
    Euler()

Sets up an Euler timestepper.
"""
struct Euler end

@inline function (::Euler)(u⃗, u⃗⁻, Δt, args...)
    return Δt * u⃗
end

@inline stages(::RK3) = 1:3
@inline stages(::Euler) = 1:1

@kernel function step_nodes!(accelerations, old_accelerations, velocities, old_velocities, positions, holdfast_z, timestepper, Δt, stage)
    p, n = @index(Global, NTuple)

    @inbounds for d=1:3
        old_velocities[p, n, d] = velocities[p, n, d]
        
        velocities[p, n, d] += timestepper(accelerations[p, n, d], old_accelerations[p, n, d], Δt, stage)
        
        old_accelerations[p, n, d] = accelerations[p, n, d]

        positions[p, n, d] += timestepper(velocities[p, n, d], old_velocities[p, n, d], Δt, stage)

        if positions[p, n, 3] + holdfast_z[p] > 0.0 
            positions[p, n, 3] = - holdfast_z[p]
        end
    end
end