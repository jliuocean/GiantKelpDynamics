using KernelAbstractions, LinearAlgebra
using KernelAbstractions.Extras: @unroll
using Oceananigans.Architectures: device, arch_array
using Oceananigans.Fields: interpolate, fractional_x_index, fractional_y_index, fractional_z_index, fractional_indices
using Oceananigans.Utils: work_layout
using Oceananigans.Operators: Vᶜᶜᶜ

const rk3 = ((8//15, nothing), (5//12, -17//60), (3//4, -5//12))

# ## Create the particles
struct Nodes{X, U, L, R, N, A, V, RE, F, UM, FM, FD}
    # nodes
    x⃗::X# node positions relative to base
    u⃗::U # node velocities in rest frame of base
    l⃗₀::L # segment unstretched length
    r⃗ˢ::R # stipe radius
    n⃗ᵇ::N # number of bladdes
    A⃗ᵇ::A # area of individual blade
    V⃗ᵖ::V # volume of pneumatocysts assuming density is air so ∼ 0 kg/m³
    r⃗ᵉ::RE # effective radius to drag over

    # forces on nodes and force history
    F⃗::F
    u⃗⁻::UM
    F⃗⁻::FM
    F⃗ᴰ::FD
end

struct GiantKelp{X, Y, Z, X0, Y0, Z0, Nodes}
    # origin position and velocity
    x::X
    y::Y
    z::Z

    x₀::X0
    y₀::Y0
    z₀::Z0

    #information about nodes
    nodes::Nodes
end

@inline tension(Δx, l₀, Aᶜ, params) = Δx>l₀ && !(Δx==0.0)  ? params.k*((Δx- l₀)/l₀)^params.α*Aᶜ : 0.0

# TODO: need to reconsile the location of the velocity field being used, the drag, and the elasticity
# Currently the elasticity acts on the nodes, while the velocity field is interpolated at the midpoint
# of the segments and the drag is exerted on the water around the segments

@kernel function step_node!(properties, model, Δt, γ, ζ, params)
    p, i = @index(Global, NTuple)

    node = @inbounds properties.nodes[p]

    x⃗ = @inbounds node.x⃗[i, :]
    if i==1
        x⃗⁻ = zeros(3)
    else
        x⃗⁻ = @inbounds node.x⃗[i-1, :]
    end

    Δx = x⃗ - x⃗⁻

    x, y, z = @inbounds [properties.x[p], properties.y[p], properties.z[p]] + x⃗

    l = sqrt(dot(Δx, Δx))

    Fᴮ = @inbounds (params.ρₒ-500)*node.V⃗ᵖ[i]*[0.0, 0.0, params.g] #currently assuming kelp is nutrally buoyant except for pneumatocysts

    if Fᴮ[3] > 0 && z >= 0  # i.e. floating up not sinking, and outside of the surface
        Fᴮ[3] = 0.0
    end

    Vᵐ = π*node.r⃗ˢ[i]^2*l + node.n⃗ᵇ[i]*node.A⃗ᵇ[i]*0.01 # TODO: change thickness to some realistic thing
    mᵉ = (Vᵐ + params.Cᵃ*(Vᵐ + node.V⃗ᵖ[i]))*params.ρₒ

    u⃗ʷ = [interpolate.(values(model.velocities), x, y, z)...]
    u⃗ᵣₑₗ = u⃗ʷ - (node.u⃗[i, :])
    sᵣₑₗ = sqrt(dot(u⃗ᵣₑₗ, u⃗ᵣₑₗ))

    a⃗ʷ = [interpolate.(values(model.timestepper.Gⁿ[(:u, :v, :w)]), x, y, z)...]
    a⃗ᵣₑₗ = a⃗ʷ - @inbounds node.F⃗[i, :]./mᵉ

    θ = acos(min(1, abs(dot(u⃗ᵣₑₗ, Δx))/(sᵣₑₗ*l + eps(0.0))))
    Aˢ = @inbounds 2*node.r⃗ˢ[i]*l*abs(sin(θ)) + π*node.r⃗ˢ[i]*abs(cos(θ))

    Fᴰ = .5*params.ρₒ*(params.Cᵈˢ*Aˢ + params.Cᵈᵇ*node.n⃗ᵇ[i]*node.A⃗ᵇ[i])*sᵣₑₗ.*u⃗ᵣₑₗ

    if i==length(node.l⃗₀)
        x⃗⁺ = x⃗ - ones(3) # doesn't matter but needs to be non-zero
        Aᶜ⁺ = 0.0 # doesn't matter
        l₀⁺ = @inbounds node.l⃗₀[i] # again, doesn't matter but probs shouldn't be zero
    else
        x⃗⁺ = @inbounds node.x⃗[i+1, :]
        Aᶜ⁺ = @inbounds π*node.r⃗ˢ[i+1]^2
        l₀⁺ = @inbounds node.l⃗₀[i+1]
    end

    Aᶜ⁻ = @inbounds π*node.r⃗ˢ[i]^2
    l₀⁻ = @inbounds node.l⃗₀[i]

    Δx⃗⁻ = x⃗⁻ - x⃗
    Δx⃗⁺ = x⃗⁺ - x⃗

    Δx⁻ = sqrt(dot(Δx⃗⁻, Δx⃗⁻))
    Δx⁺ = sqrt(dot(Δx⃗⁺, Δx⃗⁺))

    T⁻ = tension(sqrt(dot(Δx⃗⁻, Δx⃗⁻)), l₀⁻, Aᶜ⁻, params).*Δx⃗⁻./(Δx⁻+eps(0.0))
    T⁺ = tension(sqrt(dot(Δx⃗⁺, Δx⃗⁺)), l₀⁺, Aᶜ⁺, params).*Δx⃗⁺./(Δx⁺+eps(0.0))

    Fⁱ = params.ρₒ*(Vᵐ+node.V⃗ᵖ[i]).*(params.Cᵃ*a⃗ᵣₑₗ + a⃗ʷ)

    @inbounds begin 
        node.F⃗[i, :] = (Fᴮ + Fᴰ + T⁻ + T⁺ + Fⁱ)./mᵉ
        node.F⃗ᴰ[i, :] = Fᴰ + Fⁱ # store for back reaction onto water
        
        if any(isnan.(node.F⃗[i, :])) error("F is NaN: i=$i $(Fᴮ) .+ $(Fᴰ) .+ $(T⁻) .+ $(T⁺) at $x, $y, $z") end

        # Think its possibly reassigning the same values on top of eachother?
        #node.u⃗⁻[i, :] = node.u⃗[i, :]
        #node.u⃗[i, :] += rk3_substep(node.F⃗[i, :], node.F⃗⁻[i, :], Δt, γ, ζ)
        node.u⃗[i, :] += node.F⃗[i, :]*Δt
        #node.F⃗⁻[i, :] = node.F⃗[i, :]

        #node.x⃗[i, :] += rk3_substep(node.u⃗[i, :], node.u⃗⁻[i, :], Δt, γ, ζ)
        node.x⃗[i, :] += node.u⃗[i, :]*Δt

        if node.x⃗[i, 3] + properties.z[p] > 0.0 #given above bouyancy conditions this should never be possible (assuming a flow with zero vertical velocity at the surface, i.e. a real one)
            node.x⃗[i, 3] = -properties.z[p]
        end
    end
end

@inline function rk3_substep(u⃗, u⃗⁻, Δt, γ, ζ)
    return Δt*γ*u⃗ + Δt*ζ*u⃗⁻
end

@inline function rk3_substep(u⃗, u⃗⁻, Δt, γ, ζ::Nothing)
    return Δt*γ*u⃗
end

function kelp_dynamics!(particles, model, Δt)
    particles.properties.x .= particles.properties.x₀
    particles.properties.y .= particles.properties.y₀
    particles.properties.z .= particles.properties.z₀

    # calculate each particles node dynamics
    n_particles = length(particles)
    n_nodes = particles.parameters.n_nodes
    worksize = (n_particles, n_nodes)
    workgroup = (1, min(256, worksize[1]))

    for (γ, ζ) in rk3
        step_node_kernel! = step_node!(device(model.architecture), workgroup, worksize)
        step_node_event = step_node_kernel!(particles.properties, model, Δt, γ, ζ, particles.parameters)
        wait(step_node_event)
    end
end

struct LocalTransform{X, RZ, RX}
    x⃗₀ :: X
    R₁ :: RZ
    R₂ :: RX
end

@inline (transform::LocalTransform)(x, y, z) = transform.R₁*(transform.R₂*([x, y, z] - transform.x⃗₀))

@inline function LocalTransform(θ::Number, ϕ::Number, x⃗₀::Vector) 
    R₁ = LinearAlgebra.Givens(1, 3, cos(-ϕ), sin(-ϕ))
    R₂ = LinearAlgebra.Givens(1, 2, cos(θ), sin(θ))
    return LocalTransform(x⃗₀, R₁, R₂)
end

@kernel function node_weights!(drag_nodes, particles, grid, rᵉ, l⁺, l⁻, polar_transform, n)
    i, j, k = @index(Global, NTuple)

    x, y, z = Oceananigans.node(Center(), Center(), Center(), i, j, k, grid)

    x_, y_, z_ = polar_transform(x, y, z)
    r = sqrt(x_^2+y_^2)
    # if the last segment exerted a force here then don't exert the force here
    # questionable but ???
    #if n == 1 || @inbounds drag_nodes[i, j, k] == 0
        @inbounds drag_nodes[i, j, k] = ifelse((r<rᵉ)&(-l⁻-rᵉ<z_<l⁺+rᵉ), particles.parameters.drag_smoothing(r, abs(z_)-ifelse(z > 0, l⁺, l⁻), rᵉ), 0.0)
    #else
    #    @inbounds drag_nodes[i, j, k] = 0.0
    #end
end

@kernel function apply_drag!(Gᵘ, Gᵛ, Gʷ, drag_nodes, normalisations, particles, grid, F⃗ᴰ)
    i, j, k = @index(Global, NTuple)

    vol = Vᶜᶜᶜ(i, j, k, grid)
    inverse_effective_mass = @inbounds drag_nodes[i, j, k]/(normalisations*vol*particles.parameters.ρₒ)
    if any(isnan.(F⃗ᴰ.*inverse_effective_mass)) error("NaN from $F⃗ᴰ, $normalisations * $vol * .../$(drag_nodes[i, j, k])") end
    @inbounds begin
        Gᵘ[i, j, k] -= F⃗ᴰ[1]*inverse_effective_mass
        Gᵛ[i, j, k] -= F⃗ᴰ[2]*inverse_effective_mass
        Gʷ[i, j, k] -= F⃗ᴰ[3]*inverse_effective_mass
    end
end

function drag_water!(model)
    particles = model.particles
    Gᵘ, Gᵛ, Gʷ = model.timestepper.Gⁿ[(:u, :v, :w)]
    drag_nodes = model.auxiliary_fields.drag_nodes

    workgroup, worksize = work_layout(grid, :xyz)
    node_weights_kernel! = node_weights!(device(model.architecture), workgroup, worksize)
    apply_drag_kernel! = apply_drag!(device(model.architecture), workgroup, worksize)

    n_nodes = particles.parameters.n_nodes
    for p in 1:length(particles), n in 1:n_nodes
        properties = particles.properties
        node = @inbounds properties.nodes[p]

        # get node positions and size
        @inbounds begin
            x⃗ = node.x⃗[n, :] + [properties.x[p], properties.y[p], properties.z[p]]
            if n==1
                x⃗⁻ = [properties.x[p], properties.y[p], properties.z[p]]
            else
                x⃗⁻ = node.x⃗[n-1, :] + [properties.x[p], properties.y[p], properties.z[p]]
            end

            if n==n_nodes
                x⃗⁺ = x⃗
            else
                x⃗⁺ = node.x⃗[n+1, :] + [properties.x[p], properties.y[p], properties.z[p]]
            end
        end

        rᵉ = @inbounds node.r⃗ᵉ[n]

        Δx⃗ = x⃗⁺ - x⃗⁻
        
        if n==1
            l⁻ = sqrt(dot(node.x⃗[n, :], node.x⃗[n, :]))
        else
            l⁻ = sqrt(dot(node.x⃗[n, :] - node.x⃗[n-1, :], node.x⃗[n, :] - node.x⃗[n-1, :]))/2
        end
    
            
        if n==n_nodes
            l⁺ = sqrt(dot(node.x⃗[n, :] - node.x⃗[n-1, :], node.x⃗[n, :] - node.x⃗[n-1, :]))/2
        else
            l⁺ = sqrt(dot(node.x⃗[n+1, :] - node.x⃗[n, :], node.x⃗[n+1, :] - node.x⃗[n, :]))/2
        end

        θ = atan(Δx⃗[2]/(Δx⃗[1]+eps(0.0))) + π*0^(1 + sign(Δx⃗[1]))
        ϕ = atan(sqrt(Δx⃗[1]^2 + Δx⃗[2]^2+eps(0.0))/Δx⃗[3])

        node_weights_event = node_weights_kernel!(drag_nodes, particles, model.grid, rᵉ, l⁺, l⁻, LocalTransform(θ, ϕ, x⃗), n)
        wait(node_weights_event)

        normalisation = sum(drag_nodes)
        F⃗ᴰ = particles.properties.nodes[p].F⃗ᴰ[n, :]

        # fallback if nodes are closer together than gridpoints and the line joining them is parallel to a grid Axis
        # as this means there are then no nodes in the stencil. This is mainly an issue for nodes close together lying on the surface
        # As long as the (relaxed) segment lengths are properly considered this shouldn't be an issue except during startup where upstream 
        # elements will quickly move towards dowmnstream elements
        if normalisation == 0.0
            (ϵ, i), (η, j), (ζ, k) = modf.(fractional_indices(x⃗₀..., (Center(), Center(), Center()), model.grid))
            i, j, k = floor.(Int, (i, j, k))
            vol = Vᶜᶜᶜ(i, j, k, model.grid)
            inverse_effective_mass = @inbounds 1/(vol*particles.parameters.ρₒ)
            Gᵘ[i, j, k] -= F⃗ᴰ[1]*inverse_effective_mass
            Gᵛ[i, j, k] -= F⃗ᴰ[2]*inverse_effective_mass
            Gʷ[i, j, k] -= F⃗ᴰ[3]*inverse_effective_mass

            @warn "Used fallback drag application as stencil found no nodes, this should be concerning if not in the initial transient response"
        else
            apply_drag_event = apply_drag_kernel!(Gᵘ, Gᵛ, Gʷ, drag_nodes, normalisation, particles, grid, F⃗ᴰ)
            wait(apply_drag_event)
        end
    end
end
