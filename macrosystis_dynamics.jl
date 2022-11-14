using KernelAbstractions, LinearAlgebra
using KernelAbstractions.Extras: @unroll
using Oceananigans.Architectures: device, arch_array
using Oceananigans.Fields: interpolate, fractional_x_index, fractional_y_index, fractional_z_index, fractional_indices
using Oceananigans.Utils: work_layout
using Oceananigans.Operators: Vᶜᶜᶜ

const rk3 = ((8//15, nothing), (5//12, -17//60), (3//4, -5//12))

# ## Create the particles
struct Nodes
    # nodes
    x⃗# node positions relative to base
    u⃗ # node velocities in rest frame of base
    l⃗₀ # segment unstretched length
    r⃗ˢ # stipe radius
    n⃗ᵇ # number of bladdes
    A⃗ᵇ # area of individual blade
    V⃗ᵖ # volume of pneumatocysts assuming density is air so ∼ 0 kg/m³
    r⃗ᵉ # effective radius to drag over

    # forces on nodes and force history
    F⃗
    u⃗⁻
    F⃗⁻
    F⃗ᴰ
end

struct GiantKelp
    # origin position and velocity
    x
    y
    z

    x₀
    y₀
    z₀

    #information about nodes
    nodes::Nodes
end

@inline tension(Δx, l₀, Aᶜ, params) = Δx>l₀ && !(Δx==0.0)  ? params.k*((Δx- l₀)/l₀)^params.α*Aᶜ : 0.0

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

    x, y, z = @inbounds [properties.x[p], properties.y[p], properties.z[p]] + x⃗ - Δx./2

    l = sqrt(dot(Δx, Δx))

    Fᴮ = @inbounds (params.ρₒ-500)*node.V⃗ᵖ[i]*[0.0, 0.0, params.g] #currently assuming kelp is nutrally buoyant except for pneumatocysts

    if Fᴮ[3] > 0 && z >= 0  # i.e. floating up not sinking, and outside of the surface
        Fᴮ[3] = 0.0
    end

    Vᵐ = π*node.r⃗ˢ[i]*node.r⃗ˢ[i] + node.A⃗ᵇ[i]*0.01 # TODO: change thickness to some realistic thing
    mᵉ = (Vᵐ + params.Cᵃ*(Vᵐ + node.V⃗ᵖ[i]))*params.ρₒ

    u⃗ʷ = [interpolate.(values(model.velocities), x, y, z)...]
    u⃗ᵣₑₗ = u⃗ʷ - node.u⃗[i, :]
    sᵣₑₗ = sqrt(dot(u⃗ᵣₑₗ, u⃗ᵣₑₗ))

    a⃗ʷ = [interpolate.(values(model.timestepper.Gⁿ[(:u, :v, :w)]), x, y, z)...]
    a⃗ᵣₑₗ = a⃗ʷ - @inbounds node.F⃗[i, :]./mᵉ

    Aˢ = @inbounds 2*node.r⃗ˢ[i]*l*sin(acos(min(1, abs(dot(u⃗ᵣₑₗ, Δx))/(sᵣₑₗ*l + eps(0.0)))))

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
    n_nodes = length(particles.properties.nodes[1].l⃗₀)
    worksize = (n_particles, n_nodes)
    workgroup = (1, min(256, worksize[1]))

    for (γ, ζ) in rk3
        step_node_kernel! = step_node!(device(model.architecture), workgroup, worksize)
        step_node_event = step_node_kernel!(particles.properties, model, Δt, γ, ζ, particles.parameters)
        wait(step_node_event)
    end
end

function segment_polar_frame(x, y, z, x⃗, x⃗⁻)
    Δx⃗ = x⃗ - x⃗⁻
    x⃗₀ = x⃗⁻ + Δx⃗./2
    θ = atan(Δx⃗[2]/(Δx⃗[1]+eps(0.0)))
    ϕ = atan(Δx⃗[3]/sqrt(Δx⃗[1]^2 + Δx⃗[2]^2+eps(0.0)))
    Gˣʸ = LinearAlgebra.Givens(1, 2, cos(θ), sin(θ))
    Gˣᶻ = LinearAlgebra.Givens(1, 3, cos(ϕ-π/2), sin(ϕ-π/2))

    x⃗_node = Gˣᶻ*(Gˣʸ*([x, y, z] - x⃗₀))

    r = sqrt(x⃗_node[1]^2 + x⃗_node[2]^2)

    return r, x⃗_node[3]
end

@kernel function node_weights!(drag_weights, particles, grid, p, n, params)
    i, j, k = @index(Global, NTuple)

    properties = particles.properties
    node = @inbounds properties.nodes[p]

    # get node positions
    x⃗ = @inbounds node.x⃗[n, :] + [properties.x[p], properties.y[p], properties.z[p]]
    if n==1
        x⃗⁻ = @inbounds [properties.x[p], properties.y[p], properties.z[p]]
    else
        x⃗⁻ = @inbounds node.x⃗[n-1, :] + [properties.x[p], properties.y[p], properties.z[p]]
    end
    Δx⃗ = x⃗ - x⃗⁻
    x, y, z_ = Oceananigans.node(Center(), Center(), Center(), i, j, k, grid)
    r, z = segment_polar_frame(x, y, z_, x⃗, x⃗⁻)

    rᵉ = @inbounds node. r⃗ᵉ[n]
    l = sqrt(dot(Δx⃗, Δx⃗))
    @inbounds drag_weights[p, n][i, j, k] = ifelse((r<4.746*rᵉ)&(abs(z)<l/2), exp(-r^2/(2*rᵉ^2)), 0.0)
end

@kernel function calculate_normalisations!(drag_weights, weight_normalisations)
    p, n = @index(Global, NTuple)
    @inbounds weight_normalisations[p, n] = @inbounds sum(drag_weights[p, n])
end

@kernel function apply_drag!(Gᵘ, Gᵛ, Gʷ, drag_weights, normalisations, particles, grid, p, n)
    i, j, k = @index(Global, NTuple)

    vol = Vᶜᶜᶜ(i, j, k, grid)
    @inbounds begin
        F⃗ᴰ = particles.properties.nodes[p].F⃗ᴰ[n, :]
        Gᵘ[i, j, k] += F⃗ᴰ[1]*drag_weights[p, n][i, j, k]/(normalisations[p, n]*vol)
        Gᵛ[i, j, k] += F⃗ᴰ[2]*drag_weights[p, n][i, j, k]/(normalisations[p, n]*vol)
        Gʷ[i, j, k] += F⃗ᴰ[3]*drag_weights[p, n][i, j, k]/(normalisations[p, n]*vol)
    end
end

function drag_water!(model)
    particles = model.particles
    Gᵘ, Gᵛ, Gʷ = model.timestepper.Gⁿ[(:u, :v, :w)]
    drag_weights = model.auxiliary_fields.drag_weights
    normalisations = model.auxiliary_fields.drag_weight_normlisations

    workgroup, worksize = work_layout(grid, :xyz)
    node_weights_kernel! = node_weights!(device(model.architecture), workgroup, worksize)

    events = []

    for p = 1:length(particles), n = 1:length(particles.properties.nodes[1].l⃗₀)
        node_weights_event = node_weights_kernel!(drag_weights, particles, model.grid, p, n, particles.parameters)
        push!(events, node_weights_event)
    end

    wait(device(model.architecture), MultiEvent(Tuple(events)))

    n_particles = length(particles)
    n_nodes = length(particles.properties.nodes[1].l⃗₀)
    worksize_p = (n_particles, n_nodes)
    workgroup_p = (1, min(256, worksize[1]))
    
    calculate_normalisations_kernel! = calculate_normalisations!(device(model.architecture), workgroup_p, worksize_p)

    calculate_normalisations_event = calculate_normalisations_kernel!(drag_weights, normalisations)
    wait(calculate_normalisations_event)

    apply_drag_kernel! = apply_drag!(device(model.architecture), workgroup, worksize)

    events = []

    for p = 1:length(particles), n = 1:length(particles.properties.nodes[1].l⃗₀)
        apply_drag_event = apply_drag_kernel!(Gᵘ, Gᵛ, Gʷ, drag_weights, normalisations, particles, grid, p, n)
        push!(events, apply_drag_event)
    end

    wait(device(model.architecture), MultiEvent(Tuple(events)))
end
