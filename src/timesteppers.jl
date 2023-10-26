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

@kernel function step_nodes!(accelerations, old_accelerations, velocities, old_velocities, positions, timestepper, Δt, stage)
    p, n = @index(Global, NTuple)
    @inbounds begin
        old_velocities[p][n, :] .= velocities[p][n, :]
        
        velocities[p][n, :] .+= timestepper(accelerations[p][n, :], old_accelerations[p][n, :], Δt, stage)
        
        old_accelerations[p][n, :] .= accelerations[p][n, :]

        positions[p][n, :] .+= timestepper(velocities[p][n, :], old_velocities[p][n, :], Δt, stage)
    end
end