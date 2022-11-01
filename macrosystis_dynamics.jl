using KernelAbstractions, LinearAlgebra
using Oceananigans.Architectures: device, arch_array
using Oceananigans.Fields: interpolate

const rk3 = ((8//15, nothing), (5//12, -17//60), (3//4, -5//12))


# ## Create the particles
struct Nodes
    # nodes
    x⃗# node positions relative to base
    u⃗ # node velocities in rest frame of base
    ρ⃗ # each node can be different density, diagnostic of bladders and frond distribution
    V⃗ # node volume
    A⃗ᶠ # segment flow projected area
    A⃗ᶜ # cross sectional area (for elasticity)
    l⃗₀ # segment unstretched length

    # forces on nodes and force history
    F⃗
    u⃗⁻
    F⃗⁻
end
struct GiantKelp
    # origin position and velocity
    x
    y
    z
    u₀
    v₀
    w₀

    # water velocity
    u
    v
    w

    #information about nodes
    nodes::Nodes
end
@inline tension(Δx, l₀, Aᶜ, params) = Δx>l₀ && !(Δx==0.0)  ? params.k*((Δx- l₀)/l₀)^params.α*Aᶜ : 0.0

@kernel function step_node!(properties, model, Δt, γ, ζ, params)
    p, i = @index(Global, NTuple)

    node = @inbounds properties.nodes[p]

    x, y, z = [properties.x[p], properties.y[p], properties.z[p]] + node.x⃗[i, :]

    Fᴮ = @inbounds (params.ρₒ - node.ρ⃗[i])*node.V⃗[i]*[0.0, 0.0, params.g]

    if Fᴮ[3] > 0 && z >= 0  # i.e. floating up not sinking, and outside of the surface
        Fᴮ[3] = 0.0
    end

    u⃗ʷ = [interpolate.(values(model.velocities), x, y, z)...]
    u⃗ᵣₑₗ = u⃗ʷ - [properties.u₀[p], properties.v₀[p], properties.w₀[p]] - node.u⃗[i, :]

    a⃗ʷ = [interpolate.(values(model.timestepper.Gⁿ[(:u, :v, :w)]), x, y, z)...]
    a⃗ᵣₑₗ = a⃗ʷ - node.F⃗[i, :]./(node.ρ⃗[i]*node.V⃗[i] + params.ρₒ*params.Cᵃ*node.V⃗[i])

    Fᴰ = @inbounds .5*params.ρₒ*params.Cᵈ*node.A⃗ᶠ[i]*abs.(u⃗ᵣₑₗ).*u⃗ᵣₑₗ

    x⃗ = @inbounds node.x⃗[i, :]
    if i==length(node.ρ⃗)
        x⃗⁺ = x⃗ - ones(3) # doesn't matter but needs to be non-zero
        Aᶜ⁺ = 0.0 # doesn't matter
        l₀⁺ = @inbounds node.l⃗₀[i] # again, doesn't matter but probs shouldn't be zero
    else
        x⃗⁺ = @inbounds node.x⃗[i+1, :]
        Aᶜ⁺ = @inbounds node.A⃗ᶜ[i+1]
        l₀⁺ = @inbounds node.l⃗₀[i+1]
    end

    if i==1
        x⃗⁻ = zeros(3)
    else
        x⃗⁻ = @inbounds node.x⃗[i-1, :]
    end

    Aᶜ⁻ = @inbounds node.A⃗ᶜ[i]
    l₀⁻ = @inbounds node.l⃗₀[i]

    Δx⃗⁻ = x⃗⁻ - x⃗
    Δx⃗⁺ = x⃗⁺ - x⃗

    Δx⁻ = sqrt(dot(Δx⃗⁻, Δx⃗⁻))
    Δx⁺ = sqrt(dot(Δx⃗⁺, Δx⃗⁺))

    T⁻ = tension(sqrt(dot(Δx⃗⁻, Δx⃗⁻)), l₀⁻, Aᶜ⁻, params).*Δx⃗⁻./(Δx⁻+eps(0.0))
    T⁺ = tension(sqrt(dot(Δx⃗⁺, Δx⃗⁺)), l₀⁺, Aᶜ⁺, params).*Δx⃗⁺./(Δx⁺+eps(0.0))

    Fⁱ = params.ρₒ*node.V⃗[i].*(params.Cᵃ*a⃗ᵣₑₗ + a⃗ʷ)

    @inbounds begin 
        node.F⃗[i, :] = (Fᴮ + Fᴰ + T⁻ + T⁺ + Fⁱ)./(node.ρ⃗[i]*node.V⃗[i] + params.ρₒ*params.Cᵃ*node.V⃗[i])
        if any(isnan.(node.F⃗[i, :])) error("F is NaN: i=$i $(Fᴮ) .+ $(Fᴰ) .+ $(T⁻) .+ $(T⁺)") end

        node.u⃗⁻[i, :] = node.u⃗[i, :]
        node.u⃗[i, :] += rk3_substep(node.F⃗[i, :], node.F⃗⁻[i, :], Δt, γ, ζ)
        node.F⃗⁻[i, :] = node.F⃗[i, :]

        node.x⃗[i, :] += rk3_substep(node.u⃗[i, :], node.u⃗⁻[i, :], Δt, γ, ζ)

        if node.x⃗[i, 3] + properties.z[p] > 0.0 #given above bouyancy conditions this should never be possible (assuming a flow with zero vertical velocity at the surface, i.e. a real one)
            node.x⃗[i, 3] = -properties.z[p]
        end
    end
end

@inline function rk3_substep(u⃗, u⃗⁻, Δt, γ, ζ)
    @info "somehow called"
    return Δt*γ*u⃗ + Δt*ζ*u⃗⁻
end

@inline function rk3_substep(u⃗, u⃗⁻, Δt, γ, ζ::Nothing)
    return Δt*γ*u⃗
end

function dynamics!(particles, model, Δt)
    # for now, zero the base position and velocity
    particles.properties.x .= model.grid.xᶜᵃᵃ[8]
    particles.properties.y .= model.grid.yᵃᶜᵃ[8]
    particles.properties.z .= model.grid.zᵃᵃᶜ[1]

    particles.properties.u₀ .= 0.0
    particles.properties.v₀ .= 0.0
    particles.properties.w₀ .= 0.0

    # calculate each particles node dynamics
    n_particles = length(particles)
    worksize = n_particles
    workgroup = min(worksize, 256)

    for vel in (:u, :v, :w)
        tracer = model.velocities[vel]
        
        particle_property = getproperty(particles.properties, vel)

        LX, LY, LZ = location(tracer)

        update_field_property_kernel! = Oceananigans.LagrangianParticleTracking.update_field_property!(device(model.architecture), workgroup, worksize)
        source_event = update_field_property_kernel!(particle_property, particles.properties, model.grid, tracer, LX(), LY(), LZ())
        wait(source_event)
    end

    n_nodes = length(particles.properties.nodes[1].ρ⃗)
    worksize = (n_particles, n_nodes)
    workgroup = (1, min(256, worksize[1]))

    for (γ, ζ) in rk3
        step_node_kernel! = step_node!(device(model.architecture), workgroup, worksize)
        step_node_event = step_node_kernel!(particles.properties, model, Δt, 1, nothing, particles.parameters)
        wait(step_node_event)
    end
end