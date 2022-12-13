using KernelAbstractions, LinearAlgebra
using KernelAbstractions.Extras: @unroll
using Oceananigans.Architectures: device, arch_array
using Oceananigans.Fields: interpolate, fractional_x_index, fractional_y_index, fractional_z_index, fractional_indices
using Oceananigans.Utils: work_layout
using Oceananigans.Operators: Vᶜᶜᶜ
using Oceananigans: CPU

const rk3 = ((8//15, nothing), (5//12, -17//60), (3//4, -5//12))

function x⃗₀(number, depth, l₀)
    x = zeros(number, 3)
    for i in 1:number
        if l₀ * i - depth < 0
            x[i, 3] = l₀ * i
        else
            x[i, :] = [l₀ * i - depth, 0.0, depth]
        end
    end
    return x
end

# ## Create the particles
struct Nodes{VF, SF, SI}
    # nodes
    x⃗::VF# node positions relative to base
    u⃗::VF # node velocities in rest frame of base
    l⃗₀::SF # segment unstretched length
    r⃗ˢ::SF # stipe radius
    n⃗ᵇ::SI # number of bladdes
    A⃗ᵇ::SF # area of individual blade
    V⃗ᵖ::SF # volume of pneumatocysts assuming density is air so ∼ 0 kg/m³
    r⃗ᵉ::SF # effective radius to drag over

    # forces on nodes and force history
    F⃗::VF
    u⃗⁻::VF
    F⃗⁻::VF
    F⃗ᴰ::VF

    function Nodes(; number :: IT,
                     depth :: FT,
                     l₀ :: FT = 0.6,
                     x⃗ :: VF = x⃗₀(number, depth, l₀),
                     u⃗ :: VF= zeros(Float64, number, 3),
                     l⃗₀ :: SF = 0.6 * ones(number),
                     r⃗ˢ :: SF = 0.03 * ones(number),
                     n⃗ᵇ :: SI = [i*50/number for i in 1:number],
                     A⃗ᵇ :: SF = 0.1 * ones(number),
                     V⃗ᵖ :: SF = 0.05 * ones(number),
                     r⃗ᵉ :: SF = 0.5 * ones(number),
                     architecture = CPU()) where {IT, FT, VF, SF, SI}

        x⃗ = arch_array(architecture, x⃗)
        u⃗ = arch_array(architecture, u⃗)

        VFF = typeof(x⃗)

        l⃗₀ = arch_array(architecture, l⃗₀)
        r⃗ˢ = arch_array(architecture, r⃗ˢ)
        n⃗ᵇ = arch_array(architecture, n⃗ᵇ)
        A⃗ᵇ = arch_array(architecture, A⃗ᵇ)
        V⃗ᵖ = arch_array(architecture, V⃗ᵖ)
        r⃗ᵉ = arch_array(architecture, r⃗ᵉ)

        SFF = typeof(l⃗₀)
        SIF = typeof(n⃗ᵇ)

        F⃗ = zeros(FT, number, 3)#arch_array(architecture, zeros(number, 3))
        u⃗⁻ = zeros(FT, number, 3)#arch_array(architecture, zeros(number, 3))
        F⃗⁻ = zeros(FT, number, 3)#arch_array(architecture, zeros(number, 3))
        F⃗ᴰ = zeros(FT, number, 3)#arch_array(architecture, zeros(number, 3))
        return new{VFF, SFF, SIF}(x⃗, u⃗, l⃗₀, r⃗ˢ, n⃗ᵇ, A⃗ᵇ, V⃗ᵖ, r⃗ᵉ, F⃗, u⃗⁻, F⃗⁻, F⃗ᴰ)    
    end
end

struct GiantKelp{FT, N}
    # origin position and velocity
    x::FT
    y::FT
    z::FT

    x₀::FT
    y₀::FT
    z₀::FT

    scalefactor::FT

    #information about nodes
    nodes::N

    function GiantKelp(;x₀::FT, y₀::FT, z₀::FT,
                        scalefactor::FT = 1.0,
                        nodes::N = Nodes(number = 8, depth = 8.0, l₀ = 0.6),
                        architecture = CPU()) where {FT, N}

        return new{FT, N}(x₀, y₀, z₀, x₀, y₀, z₀, scalefactor, nodes)
    end
end

@inline tension(Δx, l₀, Aᶜ, params) = Δx>l₀ && !(Δx==0.0)  ? params.k*((Δx- l₀)/l₀)^params.α*Aᶜ : 0.0

# TODO: need to reconsile the location of the velocity field being used, the drag, and the elasticity
# Currently the elasticity acts on the nodes, while the velocity field is interpolated at the midpoint
# of the segments and the drag is exerted on the water around the segments

@kernel function step_node!(properties, model, Δt, γ, ζ, params)
    p, i = @index(Global, NTuple)

    node = @inbounds properties.nodes[p]

    x⃗ = @inbounds node.x⃗[i, :]
    u⃗ = @inbounds node.u⃗[i, :]
    if i==1
        x⃗⁻ = zeros(3)
        u⃗⁻ = zeros(3)
    else
        x⃗⁻ = @inbounds node.x⃗[i-1, :]
        u⃗⁻ = @inbounds node.u⃗[i-1, :]
    end

    Δx = x⃗ - x⃗⁻

    x, y, z = @inbounds [properties.x[p], properties.y[p], properties.z[p]] + x⃗

    l = sqrt(dot(Δx, Δx))

    Fᴮ = @inbounds (params.ρₒ - 500) * node.V⃗ᵖ[i] * [0.0, 0.0, params.g] #currently assuming kelp is nutrally buoyant except for pneumatocysts

    if Fᴮ[3] > 0 && z >= 0  # i.e. floating up not sinking, and outside of the surface
        Fᴮ[3] = 0.0
    end

    Vᵐ = π * node.r⃗ˢ[i] ^ 2 * l + node.n⃗ᵇ[i] * node.A⃗ᵇ[i] * 0.01 # TODO: change thickness to some realistic thing
    mᵉ = (Vᵐ + params.Cᵃ * (Vᵐ + node.V⃗ᵖ[i])) * params.ρₒ

    u⃗ʷ = [interpolate.(values(model.velocities), x, y, z)...]
    u⃗ᵣₑₗ = u⃗ʷ - (node.u⃗[i, :])
    sᵣₑₗ = sqrt(dot(u⃗ᵣₑₗ, u⃗ᵣₑₗ))

    a⃗ʷ = [interpolate.(values(model.timestepper.Gⁿ[(:u, :v, :w)]), x, y, z)...]
    a⃗ᵣₑₗ = a⃗ʷ - @inbounds node.F⃗[i, :] ./ mᵉ

    θ = acos(min(1, abs(dot(u⃗ᵣₑₗ, Δx)) / (sᵣₑₗ * l + eps(0.0))))
    Aˢ = @inbounds 2 * node.r⃗ˢ[i] * l * abs(sin(θ)) + π*node.r⃗ˢ[i] * abs(cos(θ))

    Fᴰ = .5 * params.ρₒ * (params.Cᵈˢ * Aˢ + params.Cᵈᵇ * node.n⃗ᵇ[i] * node.A⃗ᵇ[i]) * sᵣₑₗ .* u⃗ᵣₑₗ

    if i==length(node.l⃗₀)
        x⃗⁺ = x⃗ - ones(3) # doesn't matter but needs to be non-zero
        u⃗⁺ = zeros(3) # doesn't matter
        Aᶜ⁺ = 0.0 # doesn't matter
        l₀⁺ = @inbounds node.l⃗₀[i] # again, doesn't matter but probs shouldn't be zero
    else
        x⃗⁺ = @inbounds node.x⃗[i+1, :]
        u⃗⁺ = @inbounds node.u⃗[i+1, :]
        Aᶜ⁺ = @inbounds π*node.r⃗ˢ[i+1] ^ 2
        l₀⁺ = @inbounds node.l⃗₀[i+1]
    end

    Aᶜ⁻ = @inbounds π*node.r⃗ˢ[i] ^ 2
    l₀⁻ = @inbounds node.l⃗₀[i]

    Δx⃗⁻ = x⃗⁻ - x⃗
    Δx⃗⁺ = x⃗⁺ - x⃗

    Δu⃗⁻ = u⃗⁻ - u⃗
    Δu⃗⁺ = u⃗⁺ - u⃗


    l⁻ = sqrt(dot(Δx⃗⁻, Δx⃗⁻))
    l⁺ = sqrt(dot(Δx⃗⁺, Δx⃗⁺))

    T⁻ = tension(l⁻, l₀⁻, Aᶜ⁻, params) .* Δx⃗⁻ ./ (l⁻+eps(0.0)) + ifelse(l⁻ > l₀⁻, params.kᵈ * Δu⃗⁻, zeros(3))
    T⁺ = tension(l⁺, l₀⁺, Aᶜ⁺, params) .* Δx⃗⁺ ./ (l⁺+eps(0.0)) + ifelse(l⁺ > l₀⁺, params.kᵈ * Δu⃗⁺, zeros(3))

    Fⁱ = params.ρₒ * (Vᵐ+node.V⃗ᵖ[i]) .* (params.Cᵃ * a⃗ᵣₑₗ + a⃗ʷ)

    @inbounds begin 
        node.F⃗[i, :] = (Fᴮ + Fᴰ + T⁻ + T⁺ + Fⁱ) ./ mᵉ
        node.F⃗ᴰ[i, :] = Fᴰ + Fⁱ # store for back reaction onto water
        
        if any(isnan.(node.F⃗[i, :])) error("F is NaN: i=$i $(Fᴮ) .+ $(Fᴰ) .+ $(T⁻) .+ $(T⁺) at $x, $y, $z") end

        # Think its possibly reassigning the same values on top of eachother?
        node.u⃗⁻[i, :] .= node.u⃗[i, :]
        node.u⃗[i, :] .+= rk3_substep(node.F⃗[i, :], node.F⃗⁻[i, :], Δt, γ, ζ)
        #node.u⃗[i, :] += node.F⃗[i, :]*Δt
        node.F⃗⁻[i, :] .= node.F⃗[i, :]

        node.x⃗[i, :] .+= rk3_substep(node.u⃗[i, :], node.u⃗⁻[i, :], Δt, γ, ζ)
        #node.x⃗[i, :] += node.u⃗[i, :]*Δt

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

    #for substep in 1:1
        for (γ, ζ) in rk3
        #γ, ζ = 1.0, 1.0
            step_node_kernel! = step_node!(device(model.architecture), workgroup, worksize)
            step_node_event = step_node_kernel!(particles.properties, model, Δt/10, γ, ζ, particles.parameters)
            wait(step_node_event)
        end
    #end
end

struct LocalTransform{X, RZ, RX, RN, RP}
    x⃗₀ :: X
    R₁ :: RZ
    R₂ :: RX
    R₃⁺ :: RP
    R₃⁻ :: RN
end

@inline function (transform::LocalTransform)(x, y, z)
    x⃗_ = transform.R₁*(transform.R₂*([x, y, z] - transform.x⃗₀))

    if x⃗_[3]>=0.0
        x_, y_, z_ = transform.R₃⁺*x⃗_
    else
        x_, y_, z_ = transform.R₃⁻*x⃗_
    end

    return x_, y_, z_
end


@inline function LocalTransform(θ::Number, ϕ::Number, θ⁺::Number, θ⁻::Number, x⃗₀::Vector) 
    R₁ = LinearAlgebra.Givens(1, 3, cos(-ϕ), sin(-ϕ))
    R₂ = LinearAlgebra.Givens(1, 2, cos(θ), sin(θ))

    R₃⁺ = LinearAlgebra.Givens(1, 3, cos(θ⁺), sin(θ⁺))
    R₃⁻ = LinearAlgebra.Givens(1, 3, cos(θ⁻), sin(θ⁻))
    return LocalTransform(x⃗₀, R₁, R₂, R₃⁺, R₃⁻)
end

@kernel function node_weights!(drag_nodes, particles, grid, rᵉ, l⁺, l⁻, polar_transform, n)
    i, j, k = @index(Global, NTuple)

    x, y, z = Oceananigans.node(Center(), Center(), Center(), i, j, k, grid)

    x_, y_, z_ = polar_transform(x, y, z)
    r = sqrt(x_^2+y_^2)
    @inbounds drag_nodes[i, j, k] = ifelse((r<rᵉ)&(-l⁻<z_<l⁺), particles.parameters.drag_smoothing(r, rᵉ), drag_nodes[i, j, k])
end

@kernel function apply_drag!(Gᵘ, Gᵛ, Gʷ, drag_nodes, normalisations, particles, grid, F⃗ᴰ, scalefactor)
    i, j, k = @index(Global, NTuple)

    vol = Vᶜᶜᶜ(i, j, k, grid)
    inverse_effective_mass = @inbounds drag_nodes[i, j, k]/(normalisations*vol*particles.parameters.ρₒ)
    if any(isnan.(F⃗ᴰ.*inverse_effective_mass)) error("NaN from $F⃗ᴰ, $normalisations * $vol * .../$(drag_nodes[i, j, k])") end
    @inbounds begin
        Gᵘ[i, j, k] -= F⃗ᴰ[1] * inverse_effective_mass * scalefactor
        Gᵛ[i, j, k] -= F⃗ᴰ[2] * inverse_effective_mass * scalefactor
        Gʷ[i, j, k] -= F⃗ᴰ[3] * inverse_effective_mass * scalefactor
    end
end

@kernel function drag_node!(particles, properties, grid, drag_nodes, n_nodes, Gᵘ, Gᵛ, Gʷ, node_weights_kernel!, apply_drag_kernel!)
    p, n = @index(Global, NTuple)
    node = @inbounds properties.nodes[p]
    scalefactor = properties.scalefactor[p]

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
    
    θ = atan(Δx⃗[2]/(Δx⃗[1]+eps(0.0))) + π*0^(1 + sign(Δx⃗[1]))
    ϕ = atan(sqrt(Δx⃗[1]^2 + Δx⃗[2]^2+eps(0.0))/Δx⃗[3])

    cosθ⁻ = dot(Δx⃗, x⃗ - x⃗⁻)/(sqrt(dot(Δx⃗, Δx⃗))*sqrt(dot(x⃗ - x⃗⁻, x⃗ - x⃗⁻)))
    θ⁻ = -1.0<=cosθ⁻<=1.0 ? acos(cosθ⁻) : 0.0

    if n==n_nodes
        l⁺ = sqrt(dot(node.x⃗[n, :] - node.x⃗[n-1, :], node.x⃗[n, :] - node.x⃗[n-1, :]))/2
        θ⁺ = θ⁻
    else
        l⁺ = sqrt(dot(node.x⃗[n+1, :] - node.x⃗[n, :], node.x⃗[n+1, :] - node.x⃗[n, :]))/2
        cosθ⁺ = - dot(Δx⃗, x⃗⁺ - x⃗)/(sqrt(dot(Δx⃗, Δx⃗))*sqrt(dot(x⃗⁺ - x⃗, x⃗⁺ - x⃗)))
        θ⁺ = -1.0<=cosθ⁺<=1.0 ? acos(cosθ⁺) : 0.0
    end

    node_weights_event = node_weights_kernel!(drag_nodes, particles, grid, rᵉ, l⁺, l⁻, LocalTransform(θ, ϕ, θ⁺, θ⁻, x⃗), n)
    wait(node_weights_event)

    normalisation = sum(drag_nodes)
    F⃗ᴰ = node.F⃗ᴰ[n, :]

    # fallback if nodes are closer together than gridpoints and the line joining them is parallel to a grid Axis
    # as this means there are then no nodes in the stencil. This is mainly an issue for nodes close together lying on the surface
    # As long as the (relaxed) segment lengths are properly considered this shouldn't be an issue except during startup where upstream 
    # elements will quickly move towards dowmnstream elements
    if normalisation == 0.0
        (ϵ, i), (η, j), (ζ, k) = modf.(fractional_indices(x⃗..., (Center(), Center(), Center()), model.grid))
        i, j, k = floor.(Int, (i, j, k))
        vol = Vᶜᶜᶜ(i, j, k, model.grid)
        inverse_effective_mass = @inbounds 1/(vol*particles.parameters.ρₒ)
        Gᵘ[i, j, k] -= F⃗ᴰ[1] * inverse_effective_mass * scalefactor
        Gᵛ[i, j, k] -= F⃗ᴰ[2] * inverse_effective_mass * scalefactor
        Gʷ[i, j, k] -= F⃗ᴰ[3] * inverse_effective_mass * scalefactor

        @warn "Used fallback drag application as stencil found no nodes, this should be concerning if not in the initial transient response at $p, $n"
    else
        apply_drag_event = apply_drag_kernel!(Gᵘ, Gᵛ, Gʷ, drag_nodes, normalisation, particles, grid, F⃗ᴰ, scalefactor)
        wait(apply_drag_event)
    end
end

function drag_water!(model)
    particles = model.particles
    Gᵘ, Gᵛ, Gʷ = model.timestepper.Gⁿ[(:u, :v, :w)]
    drag_nodes = model.auxiliary_fields.drag_nodes

    workgroup, worksize = work_layout(grid, :xyz)
    node_weights_kernel! = node_weights!(device(model.architecture), workgroup, worksize)
    apply_drag_kernel! = apply_drag!(device(model.architecture), workgroup, worksize)

    n_particles = length(particles)
    n_nodes = particles.parameters.n_nodes

    drag_water_node_kernel! = drag_node!(device(model.architecture), (1, min(256, n_particles)), (n_particles, n_nodes))

    drag_nodes_event = drag_water_node_kernel!(particles, particles.properties, model.grid, drag_nodes, n_nodes, Gᵘ, Gᵛ, Gʷ, node_weights_kernel!, apply_drag_kernel!)
    wait(drag_nodes_event)
end
