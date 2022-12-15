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

struct Euler end

@inline function (::Euler)(u⃗, u⃗⁻, Δt, args...)
    return Δt * u⃗
end

stages(::RK3) = 1:3
stages(::Euler) = 1:1