using KernelAbstractions, LinearAlgebra
using KernelAbstractions.Extras: @unroll
using Oceananigans.Architectures: device, arch_array
using Oceananigans.Fields: interpolate, fractional_x_index, fractional_y_index, fractional_z_index, fractional_indices
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
    u₀
    v₀
    w₀

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
    u⃗ᵣₑₗ = u⃗ʷ - [properties.u₀[p], properties.v₀[p], properties.w₀[p]] - node.u⃗[i, :]
    sᵣₑₗ = sqrt(dot(u⃗ᵣₑₗ, u⃗ᵣₑₗ))

    a⃗ʷ = [interpolate.(values(model.timestepper.Gⁿ[(:u, :v, :w)]), x, y, z)...]
    a⃗ᵣₑₗ = a⃗ʷ - @inbounds node.F⃗[i, :]./mᵉ

    Aˢ = @inbounds 2*node.r⃗ˢ[i]*l*sin(acos(min(1, abs(dot(u⃗ᵣₑₗ, Δx))/(sᵣₑₗ*l))))
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
    # for now, zero the base position and velocity
    particles.properties.x .= model.grid.xᶜᵃᵃ[1]
    particles.properties.y .= model.grid.yᵃᶜᵃ[8]
    particles.properties.z .= model.grid.zᵃᵃᶜ[1]

    particles.properties.u₀ .= 0.0
    particles.properties.v₀ .= 0.0
    particles.properties.w₀ .= 0.0

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

@kernel function drag_water_node!(properties, model, params)
    p, i = @index(Global, NTuple)

    node = @inbounds properties.nodes[p]

    # get node positions
    x⃗ = @inbounds node.x⃗[i, :] + [properties.x[p], properties.y[p], properties.z[p]]
    if i==1
        x⃗⁻ = @inbounds [properties.x[p], properties.y[p], properties.z[p]]
    else
        x⃗⁻ = @inbounds node.x⃗[i-1, :] + [properties.x[p], properties.y[p], properties.z[p]]
    end

    # change to i, j, k
    r⃗ = [fractional_indices(x⃗..., (Center(), Center(), Center()), model.grid)...]
    r⃗⁻ = [fractional_indices(x⃗⁻..., (Center(), Center(), Center()), model.grid)...]

    # effective radius of drag in i,j, k units, wrong if inhomogeneous grid
    rᵈ = node. r⃗ᵉ[i]/grid.Δxᶜᵃᵃ

    # work out how many points the drag is exerted on to get the mass to divide by
    mᵈ = 0.0
    for i=1:grid.Nx, j=1:grid.Ny, k=1:grid.Nz if inside_cylinder(r⃗, r⃗⁻, rᵈ, i, j, k)
        mᵈ += params.ρₒ*Oceananigans.Operators.Vᶜᶜᶜ(i, j, k, grid)
    end end

    # apply the drag to the tendencies 
    # Think I either have to itterate twice or have three new fields per node to add the tendencies to, and then divide by the mass after?
    F⃗ᴰ = node.F⃗ᴰ[i, :]
    for i=1:grid.Nx, j=1:grid.Ny, k=1:grid.Nz if inside_cylinder(r⃗, r⃗⁻, rᵈ, i, j, k)
        model.timestepper.Gⁿ.u[i, j, k] -= F⃗ᴰ[1]/mᵈ
        model.timestepper.Gⁿ.v[i, j, k] -= F⃗ᴰ[2]/mᵈ
        model.timestepper.Gⁿ.w[i, j, k] -= F⃗ᴰ[3]/mᵈ
    end end
end

@inline function inside_cylinder(r⃗, r⃗⁻, rᵈ, i, j, k)
    n⃗ = [i, j, k]
    Δr⃗ = r⃗⁻ - r⃗
    nΔr⃗ᵐⁱⁿ = cross(Δr⃗, n⃗ - r⃗⁻)/sqrt(dot(Δr⃗, Δr⃗))
    rᵐⁱⁿ = sqrt(dot(nΔr⃗ᵐⁱⁿ, nΔr⃗ᵐⁱⁿ))

    return rᵐⁱⁿ <= rᵈ
end

function drag_water!(model, Δt)
    particles = model.particles

    # calculate each particles node dynamics
    n_particles = length(particles)
    worksize = n_particles
    workgroup = min(worksize, 256)
    
    n_particles = length(particles)
    n_nodes = length(particles.properties.nodes[1].l⃗₀)
    worksize = (n_particles, n_nodes)
    workgroup = (1, min(256, worksize[1]))
    
    drag_kernel! = drag_water_node!(device(model.architecture), workgroup, worksize)
    drag_event = drag_kernel!(particles.properties, model, particles.parameters)
    wait(drag_event)
end