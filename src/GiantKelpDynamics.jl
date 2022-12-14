module GiantKelpDynamics

export Nodes, GiantKelp, kelp_dynamics!, drag_water!

using KernelAbstractions, LinearAlgebra
using KernelAbstractions.Extras: @unroll
using Oceananigans.Architectures: device, arch_array
using Oceananigans.Fields: interpolate, fractional_x_index, fractional_y_index, fractional_z_index, fractional_indices
using Oceananigans.Utils: work_layout
using Oceananigans.Operators: Vᶜᶜᶜ
using Oceananigans: CPU, node, Center, CenterField

const rk3 = ((8//15, nothing), (5//12, -17//60), (3//4, -5//12))

function x⃗₀(number, depth, l₀, initial_stretch)
    x = zeros(number, 3)
    for i in 1:number - 1
        if l₀ * initial_stretch * i - depth < 0
            x[i, 3] = l₀ * initial_stretch * i
        else
            x[i, :] = [l₀ * initial_stretch * i - depth, 0.0, depth]
        end
    end
    if l₀ * initial_stretch * (number - 1) + l₀ - depth < 0
        x[number, 3] = l₀ * initial_stretch * (number - 1) + l₀
    else
        x[number, :] = [l₀ * initial_stretch * (number - 1) + l₀ - depth, 0.0, depth]
    end
    return x
end

struct GiantKelp{FT, VF, SF, FA}
    # origin position and velocity
    x :: FT
    y :: FT
    z :: FT

    fixed_x :: FT
    fixed_y :: FT
    fixed_z :: FT

    scalefactor::FT

    #information about nodes
    node_positions :: VF
    node_velocities :: VF 
    node_relaxed_lengths :: SF
    node_stipe_radii :: SF 
    node_blade_areas :: SF
    node_pneumatocyst_volumes :: SF # assuming density is air so ∼ 0 kg/m³
    node_effective_radii :: SF # effective radius to drag over

    # forces on nodes and force history
    node_accelerations :: VF
    node_old_velocities :: VF
    node_old_accelerations :: VF
    node_drag_forces :: VF

    drag_fields :: FA # array of drag fields for each node

    function GiantKelp(; grid, base_x::FT, base_y::FT, base_z::FT,
                         scalefactor::FT = 1.0,
                         number_nodes :: IT = 8,
                         depth :: FT = 8.0,
                         segment_unstretched_length :: FT = 0.6,
                         node_positions :: VF = x⃗₀(number_nodes, depth, segment_unstretched_length, 2.5),
                         node_velocities :: VF= zeros(Float64, number_nodes, 3),
                         node_relaxed_lengths :: SF = segment_unstretched_length * ones(number_nodes),
                         node_stipe_radii :: SF = 0.03 * ones(number_nodes),
                         node_blade_areas :: SF = 0.1 .* [i*50/number_nodes for i in 1:number_nodes],
                         node_pneumatocyst_volumes :: SF = 0.05 * ones(number_nodes),
                         node_effective_radii :: SF = 0.5 * ones(number_nodes),
                         architecture = CPU()) where {IT, FT, VF, SF}

        # some of this mess is redundant from trying to make GPU work
        node_positions = arch_array(architecture, node_positions)
        node_velocities = arch_array(architecture, node_velocities)

        VFF = typeof(node_positions) # float vector array type

        node_relaxed_lengths = arch_array(architecture, node_relaxed_lengths)
        node_stipe_radii = arch_array(architecture, node_stipe_radii)
        node_blade_areas = arch_array(architecture, node_blade_areas)
        node_pneumatocyst_volumes = arch_array(architecture, node_pneumatocyst_volumes)
        node_effective_radii = arch_array(architecture, node_effective_radii)

        SFF = typeof(node_relaxed_lengths) # float scalar array type

        node_accelerations = arch_array(architecture, zeros(FT, number_nodes, 3))
        node_old_velocities = arch_array(architecture, zeros(FT, number_nodes, 3))
        node_old_accelerations = arch_array(architecture, zeros(FT, number_nodes, 3))
        node_drag_forces = arch_array(architecture, zeros(FT, number_nodes, 3))

        drag_fields = arch_array(architecture, [CenterField(grid) for i in 1:number_nodes])
        FA = typeof(drag_fields)

        return new{FT, VFF, SFF, FA}(base_x, base_y, base_z, 
                                     base_x, base_y, base_z, 
                                     scalefactor, 
                                     node_positions, 
                                     node_velocities, 
                                     node_relaxed_lengths,
                                     node_stipe_radii,
                                     node_blade_areas, 
                                     node_pneumatocyst_volumes, 
                                     node_effective_radii, 
                                     node_accelerations, 
                                     node_old_velocities, 
                                     node_old_accelerations, 
                                     node_drag_forces,
                                     drag_fields)
    end
end

@inline tension(Δx, l₀, Aᶜ, params) = Δx > l₀ && !(Δx == 0.0)  ? params.k * ((Δx - l₀) / l₀) ^ params.α * Aᶜ : 0.0

@kernel function step_node!(x_base, y_base, z_base, 
                            node_positions, 
                            node_velocities, 
                            node_pneumatocyst_volumes, 
                            node_stipe_radii, 
                            node_blade_areas,
                            node_relaxed_lengths, 
                            node_accelerations, 
                            node_drag_forces, 
                            node_old_velocities,
                            node_old_accelerations, 
                            water_velocities, 
                            water_acceleration, 
                            Δt, γ, ζ, params)

    p, i = @index(Global, NTuple)

    x⃗ⁱ = @inbounds node_positions[p][i, :]
    u⃗ⁱ = @inbounds node_velocities[p][i, :]

    if i == 1
        x⃗⁻ = zeros(3)
        u⃗ⁱ⁻¹ = zeros(3)
    else
        x⃗⁻ = @inbounds node_positions[p][i-1, :]
        u⃗ⁱ⁻¹ = @inbounds node_velocities[p][i-1, :]
    end

    Δx⃗ = x⃗ⁱ - x⃗⁻

    x, y, z = @inbounds [x_base[p], y_base[p], z_base[p]] + x⃗ⁱ

    l = sqrt(dot(Δx⃗, Δx⃗))
    Vᵖ = node_pneumatocyst_volumes[p][i]

    Fᴮ = @inbounds (params.ρₒ - 500) * Vᵖ * [0.0, 0.0, params.g] #currently assuming kelp is nutrally buoyant except for pneumatocysts

    if Fᴮ[3] > 0 && z >= 0  # i.e. floating up not sinking, and outside of the surface
        Fᴮ[3] = 0.0
    end

    Aᵇ = node_blade_areas[p][i]
    rˢ = node_stipe_radii[p][i]
    Vᵐ = π * rˢ ^ 2 * l + Aᵇ * 0.01 # TODO: change thickness to some realistic thing
    mᵉ = (Vᵐ + params.Cᵃ * (Vᵐ + Vᵖ)) * params.ρₒ

    u⃗ʷ = [interpolate.(values(water_velocities), x, y, z)...]
    u⃗ᵣₑₗ = u⃗ʷ - u⃗ⁱ
    sᵣₑₗ = sqrt(dot(u⃗ᵣₑₗ, u⃗ᵣₑₗ))

    a⃗ʷ = [interpolate.(values(water_acceleration), x, y, z)...]
    a⃗ⁱ = node_accelerations[p][i, :]
    a⃗ᵣₑₗ = a⃗ʷ - a⃗ⁱ

    θ = acos(min(1, abs(dot(u⃗ᵣₑₗ, Δx⃗)) / (sᵣₑₗ * l + eps(0.0))))
    Aˢ = @inbounds 2 * rˢ * l * abs(sin(θ)) + π * rˢ * abs(cos(θ))

    Fᴰ = .5 * params.ρₒ * (params.Cᵈˢ * Aˢ + params.Cᵈᵇ * Aᵇ) * sᵣₑₗ .* u⃗ᵣₑₗ

    if i == length(node_relaxed_lengths[p])
        x⃗⁺ = x⃗ⁱ - ones(3) # doesn't matter but needs to be non-zero
        u⃗ⁱ⁺¹ = zeros(3) # doesn't matter
        Aᶜ⁺ = 0.0 # doesn't matter
        l₀⁺ = @inbounds node_relaxed_lengths[p][i] # again, doesn't matter but probs shouldn't be zero
    else
        x⃗⁺ = @inbounds node_positions[p][i+1, :]
        u⃗ⁱ⁺¹ = @inbounds node_velocities[p][i+1, :]
        Aᶜ⁺ = @inbounds π * node_stipe_radii[p][i+1] ^ 2
        l₀⁺ = @inbounds node_relaxed_lengths[p][i+1]
    end

    Aᶜ⁻ = @inbounds π * rˢ ^ 2
    l₀⁻ = @inbounds node_relaxed_lengths[p][i]

    Δx⃗⁻ = x⃗⁻ - x⃗ⁱ
    Δx⃗⁺ = x⃗⁺ - x⃗ⁱ

    Δu⃗ⁱ⁻¹ = u⃗ⁱ⁻¹ - u⃗ⁱ
    Δu⃗ⁱ⁺¹ = u⃗ⁱ⁺¹ - u⃗ⁱ

    l⁻ = sqrt(dot(Δx⃗⁻, Δx⃗⁻))
    l⁺ = sqrt(dot(Δx⃗⁺, Δx⃗⁺))

    T⁻ = tension(l⁻, l₀⁻, Aᶜ⁻, params) .* Δx⃗⁻ ./ (l⁻ + eps(0.0)) + ifelse(l⁻ > l₀⁻, params.kᵈ * Δu⃗ⁱ⁻¹, zeros(3))
    T⁺ = tension(l⁺, l₀⁺, Aᶜ⁺, params) .* Δx⃗⁺ ./ (l⁺ + eps(0.0)) + ifelse(l⁺ > l₀⁺, params.kᵈ * Δu⃗ⁱ⁺¹, zeros(3))

    Fⁱ = params.ρₒ * (Vᵐ + Vᵖ) .* (params.Cᵃ * a⃗ᵣₑₗ + a⃗ʷ)

    @inbounds begin 
        node_accelerations[p][i, :] .= (Fᴮ + Fᴰ + T⁻ + T⁺ + Fⁱ) ./ mᵉ
        node_drag_forces[p][i, :] .= Fᴰ + Fⁱ # store for back reaction onto water
        
        if any(isnan.(node_accelerations[p][i, :])) error("F is NaN: i=$i $(Fᴮ) .+ $(Fᴰ) .+ $(T⁻) .+ $(T⁺) at $x, $y, $z") end

        # Think its possibly reassigning the same values on top of eachother?
        node_old_velocities[p][i, :] .= node_velocities[p][i, :]
        node_velocities[p][i, :] .+= rk3_substep(node_accelerations[p][i, :], node_old_accelerations[p][i, :], Δt, γ, ζ)
        #node_velocities[p][i, :] .+= node_accelerations[p][i, :] * Δt
        node_old_accelerations[p][i, :] .= node_accelerations[p][i, :]

        node_positions[p][i, :] .+= rk3_substep(node_velocities[p][i, :], node_old_velocities[p][i, :], Δt, γ, ζ)
        #node_positions[p][i, :] += node_velocities[p][i, :] * Δt

        if node_positions[p][i, 3] + z_base[p] > 0.0 #given above bouyancy conditions this should never be possible (assuming a flow with zero vertical velocity at the surface, i.e. a real one)
            node_positions[p][i, 3] = - z_base[p]
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
    particles.properties.x .= particles.properties.fixed_x
    particles.properties.y .= particles.properties.fixed_y
    particles.properties.z .= particles.properties.fixed_z

    # calculate each particles node dynamics
    n_particles = length(particles)
    n_nodes = particles.parameters.n_nodes
    worksize = (n_particles, n_nodes)
    workgroup = (1, min(256, worksize[1]))

    n_substeps = 10
    for substep in 1:n_substeps        
        for (γ, ζ) in rk3
        #γ, ζ = 1.0, 1.0
            step_node_kernel! = step_node!(device(model.architecture), workgroup, worksize)

            step_node_event = step_node_kernel!(particles.properties.x, 
                                                particles.properties.y, 
                                                particles.properties.z, 
                                                particles.properties.node_positions, 
                                                particles.properties.node_velocities, 
                                                particles.properties.node_pneumatocyst_volumes, 
                                                particles.properties.node_stipe_radii,  
                                                particles.properties.node_blade_areas, 
                                                particles.properties.node_relaxed_lengths, 
                                                particles.properties.node_accelerations, 
                                                particles.properties.node_drag_forces, 
                                                particles.properties.node_old_velocities, 
                                                particles.properties.node_old_accelerations, 
                                                model.velocities, 
                                                model.timestepper.Gⁿ[(:u, :v, :w)], 
                                                Δt/n_substeps, γ, ζ, particles.parameters)

            wait(step_node_event)
        end
    end
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

@kernel function node_weights!(drag_field, grid, rᵉ, l⁺, l⁻, polar_transform, parameters)
    i, j, k = @index(Global, NTuple)

    x, y, z = node(Center(), Center(), Center(), i, j, k, grid)

    x_, y_, z_ = polar_transform(x, y, z)
    r = sqrt(x_ ^ 2 + y_ ^ 2)
    @inbounds drag_field[i, j, k] = ifelse((r < rᵉ) & (-l⁻ < z_ < l⁺), parameters.drag_smoothing(r, rᵉ), 0.0)
end

@kernel function apply_drag!(water_accelerations, drag_field, normalisations, grid, Fᴰ, scalefactor, parameters)
    i, j, k = @index(Global, NTuple)

    vol = Vᶜᶜᶜ(i, j, k, grid)
    inverse_effective_mass = @inbounds drag_field[i, j, k] / (normalisations * vol * parameters.ρₒ)
    if any(isnan.(Fᴰ .* inverse_effective_mass)) error("NaN from $Fᴰ, $normalisations * $vol * .../$(drag_nodes[i, j, k])") end
    @inbounds begin
        water_accelerations.u[i, j, k] -= Fᴰ[1] * inverse_effective_mass * scalefactor
        water_accelerations.v[i, j, k] -= Fᴰ[2] * inverse_effective_mass * scalefactor
        water_accelerations.w[i, j, k] -= Fᴰ[3] * inverse_effective_mass * scalefactor
    end
end

@kernel function drag_node!(base_x, base_y, base_z, 
                            scalefactors, node_positions, 
                            node_effective_radii, 
                            node_drag_forces, 
                            grid, drag_fields, 
                            n_nodes, 
                            water_accelerations, 
                            node_weights_kernel!, 
                            apply_drag_kernel!, 
                            parameters)
                            
    p, n = @index(Global, NTuple)

    scalefactor = @inbounds scalefactors[p]

    # get node positions and size
    @inbounds begin
        x⃗ = @inbounds node_positions[p][n, :] + [base_x[p], base_y[p], base_z[p]]
        if n==1
            x⃗⁻ =  [base_x[p], base_y[p], base_z[p]]
        else
            x⃗⁻ = node_positions[p][n - 1, :] + [base_x[p], base_y[p], base_z[p]]
        end

        if n == n_nodes
            x⃗⁺ = x⃗
        else
            x⃗⁺ = node_positions[p][n + 1, :] + [base_x[p], base_y[p], base_z[p]]
        end
    end

    rᵉ = @inbounds node_effective_radii[p][n]

    Δx⃗ = x⃗⁺ - x⃗⁻
        
    if n == 1
        l⁻ = sqrt(dot(node_positions[p][n, :], node_positions[p][n, :]))
    else
        l⁻ = sqrt(dot(node_positions[p][n, :] - node_positions[p][n - 1, :], node_positions[p][n, :] - node_positions[p][n - 1, :]))/2
    end
    
    θ = atan(Δx⃗[2] / (Δx⃗[1] + eps(0.0))) + π * 0 ^ (1 + sign(Δx⃗[1]))
    ϕ = atan(sqrt(Δx⃗[1] ^ 2 + Δx⃗[2] ^ 2 + eps(0.0)) / Δx⃗[3])

    cosθ⁻ = dot(Δx⃗, x⃗ - x⃗⁻) / (sqrt(dot(Δx⃗, Δx⃗)) * sqrt(dot(x⃗ - x⃗⁻, x⃗ - x⃗⁻)))
    θ⁻ = -1.0 <= cosθ⁻ <= 1.0 ? acos(cosθ⁻) : 0.0

    if n == n_nodes
        l⁺ = sqrt(dot(node_positions[p][n, :] - node_positions[p][n - 1, :], node_positions[p][n, :] - node_positions[p][n - 1, :])) / 2
        θ⁺ = θ⁻
    else
        l⁺ = sqrt(dot(node_positions[p][n + 1, :] - node_positions[p][n, :], node_positions[p][n + 1, :] - node_positions[p][n, :])) / 2
        cosθ⁺ = - dot(Δx⃗, x⃗⁺ - x⃗) / (sqrt(dot(Δx⃗, Δx⃗)) * sqrt(dot(x⃗⁺ - x⃗, x⃗⁺ - x⃗)))
        θ⁺ = -1.0 <= cosθ⁺ <= 1.0 ? acos(cosθ⁺) : 0.0
    end

    node_weights_event = @inbounds node_weights_kernel!(drag_fields[p][n], grid, rᵉ, l⁺, l⁻, LocalTransform(θ, ϕ, θ⁺, θ⁻, x⃗), parameters)
    wait(node_weights_event)

    normalisation = sum(@inbounds drag_fields[p][n])
    Fᴰ = @inbounds node_drag_forces[p][n, :]

    # fallback if nodes are closer together than gridpoints and the line joining them is parallel to a grid Axis
    # as this means there are then no nodes in the stencil. This is mainly an issue for nodes close together lying on the surface
    # As long as the (relaxed) segment lengths are properly considered this shouldn't be an issue except during startup where upstream 
    # elements will quickly move towards dowmnstream elements
    if normalisation == 0.0
        (ϵ, i), (η, j), (ζ, k) = modf.(fractional_indices(x⃗..., (Center(), Center(), Center()), grid))
        i, j, k = floor.(Int, (i, j, k))
        vol = Vᶜᶜᶜ(i, j, k, grid)
        inverse_effective_mass = @inbounds 1 / (vol * parameters.ρₒ)
        water_accelerations.u[i, j, k] -= Fᴰ[1] * inverse_effective_mass * scalefactor
        water_accelerations.v[i, j, k] -= Fᴰ[2] * inverse_effective_mass * scalefactor
        water_accelerations.w[i, j, k] -= Fᴰ[3] * inverse_effective_mass * scalefactor

        @warn "Used fallback drag application as stencil found no nodes, this should be concerning if not in the initial transient response at $p, $n"
    else
        apply_drag_event = @inbounds apply_drag_kernel!(water_accelerations, drag_fields[p][n], normalisation, grid, Fᴰ, scalefactor, parameters)
        wait(apply_drag_event)
    end
end

function drag_water!(model)
    particles = model.particles
    water_accelerations = model.timestepper.Gⁿ[(:u, :v, :w)]

    workgroup, worksize = work_layout(model.grid, :xyz)
    node_weights_kernel! = node_weights!(device(model.architecture), workgroup, worksize)
    apply_drag_kernel! = apply_drag!(device(model.architecture), workgroup, worksize)

    n_particles = length(particles)
    n_nodes = @inbounds length(particles.properties.node_relaxed_lengths[1])

    drag_water_node_kernel! = drag_node!(device(model.architecture), (1, min(256, n_particles)), (n_particles, n_nodes))

    drag_nodes_event = drag_water_node_kernel!(particles.properties.x, particles.properties.y, particles.properties.z, 
                                               particles.properties.scalefactor, 
                                               particles.properties.node_positions, 
                                               particles.properties.node_effective_radii, 
                                               particles.properties.node_drag_forces, 
                                               model.grid, 
                                               particles.properties.drag_fields, 
                                               n_nodes, 
                                               water_accelerations, 
                                               node_weights_kernel!, 
                                               apply_drag_kernel!, 
                                               particles.parameters)
    wait(drag_nodes_event)
end

end # module