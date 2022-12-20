module GiantKelpDynamics

export GiantKelp, kelp_dynamics!, drag_water!

using KernelAbstractions, LinearAlgebra, StructArrays
using KernelAbstractions.Extras: @unroll
using Oceananigans.Architectures: device, arch_array
using Oceananigans.Fields: interpolate, fractional_x_index, fractional_y_index, fractional_z_index, fractional_indices
using Oceananigans.Utils: work_layout
using Oceananigans.Operators: Vᶜᶜᶜ
using Oceananigans: CPU, node, Center, CenterField
using Oceananigans.LagrangianParticleTracking: AbstractParticle, LagrangianParticles

include("timesteppers.jl")

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

struct GiantKelp{FT, VF, SF, FA} <: AbstractParticle
    # origin position and velocity
    x :: FT
    y :: FT
    z :: FT

    fixed_x :: FT
    fixed_y :: FT
    fixed_z :: FT

    scalefactor::FT

    #information about nodes
    positions :: VF
    velocities :: VF 
    relaxed_lengths :: SF
    stipe_radii :: SF 
    blade_areas :: SF
    pneumatocyst_volumes :: SF # assuming density is air so ∼ 0 kg/m³
    effective_radii :: SF # effective radius to drag over

    # forces on nodes and force history
    accelerations :: VF
    old_velocities :: VF
    old_accelerations :: VF
    drag_forces :: VF

    drag_field :: FA # array of drag fields for each node
end

@inline guassian_smoothing(r, rᵉ) = exp(-(r) ^ 2 / (2 * rᵉ ^ 2))/sqrt(2 * π * rᵉ ^ 2)
@inline no_smoothing(r, rᵉ) = 1.0

function GiantKelp(; grid, base_x::Vector{FT}, base_y, base_z,
                      number_kelp = length(base_x),
                      scalefactor = ones(length(base_x)),
                      number_nodes = 8,
                      depth = 8.0,
                      segment_unstretched_length = 0.6,
                      initial_stipe_radii = 0.03,
                      initial_blade_areas = 0.1 .* [i*20/number_nodes for i in 1:number_nodes],
                      initial_pneumatocyst_volume = 0.05 * ones(number_nodes),
                      initial_effective_radii = 0.5 * ones(number_nodes),
                      architecture = CPU(),
                      parameters = (k = 10 ^ 5, 
                                    α = 1.41, 
                                    ρₒ = 1026.0, 
                                    ρₐ = 1.225, 
                                    g = 9.81, 
                                    Cᵈˢ = 1.0, 
                                    Cᵈᵇ= 0.4 * 12 ^ -0.485, 
                                    Cᵃ = 3.0,
                                    drag_smoothing = no_smoothing,
                                    n_nodes = number_nodes,
                                    τ = 5.0,
                                    kᵈ = 500),
                      timestepper = RK3(),
                      max_Δt = 0.5) where {FT}

    base_x = arch_array(architecture, base_x)
    base_y = arch_array(architecture, base_y)
    base_z = arch_array(architecture, base_z)
    scalefactor = arch_array(architecture, scalefactor)

    AFT = typeof(base_x)

    positions = []
    velocities = [arch_array(architecture, zeros(number_nodes, 3)) for p in 1:number_kelp]

    for i in 1:number_kelp
        push!(positions, arch_array(architecture, x⃗₀(number_nodes, depth, segment_unstretched_length, 2.5)))
    end

    VF = typeof(positions) # float vector array type

    relaxed_lengths = [arch_array(architecture, segment_unstretched_length .* ones(number_nodes)) for p in 1:number_kelp]
    stipe_radii = [arch_array(architecture, ones(number_nodes) .* initial_stipe_radii) for p in 1:number_kelp]
    blade_areas = [arch_array(architecture, ones(number_nodes) .* initial_blade_areas) for p in 1:number_kelp]
    pneumatocyst_volumes = [arch_array(architecture, ones(number_nodes) .* initial_pneumatocyst_volume) for p in 1:number_kelp]
    effective_radii = [arch_array(architecture, ones(number_nodes) .* initial_effective_radii) for p in 1:number_kelp]

    SF = typeof(relaxed_lengths) # float scalar array type

    accelerations = [arch_array(architecture, zeros(FT, number_nodes, 3)) for p in 1:number_kelp]
    old_velocities = [arch_array(architecture, zeros(FT, number_nodes, 3)) for p in 1:number_kelp]
    old_accelerations = [arch_array(architecture, zeros(FT, number_nodes, 3)) for p in 1:number_kelp]
    drag_forces = [arch_array(architecture, zeros(FT, number_nodes, 3)) for p in 1:number_kelp]

    drag_field = [CenterField(grid) for p in 1:number_kelp]

    FA = typeof(drag_field)

    kelps = StructArray{GiantKelp{AFT, VF, SF, FA}}((base_x, base_y, base_z, 
                                                     copy(base_x), copy(base_y), copy(base_z), 
                                                     scalefactor, 
                                                     positions, 
                                                     velocities, 
                                                     relaxed_lengths,
                                                     stipe_radii,
                                                     blade_areas, 
                                                     pneumatocyst_volumes, 
                                                     effective_radii, 
                                                     accelerations, 
                                                     old_velocities, 
                                                     old_accelerations, 
                                                     drag_forces,
                                                     drag_field))

    return LagrangianParticles(kelps; parameters = merge(parameters, (; timestepper, max_Δt)), dynamics = kelp_dynamics!)
end

@inline tension(Δx, l₀, Aᶜ, params) = Δx > l₀ && !(Δx == 0.0)  ? params.k * ((Δx - l₀) / l₀) ^ params.α * Aᶜ : 0.0

@kernel function step_node!(x_base, 
                            y_base, 
                            z_base, 
                            positions, 
                            velocities, 
                            pneumatocyst_volumes, 
                            stipe_radii, 
                            blade_areas,
                            relaxed_lengths, 
                            accelerations, 
                            drag_forces, 
                            old_velocities,
                            old_accelerations, 
                            water_velocities, 
                            water_accelerations, 
                            Δt, 
                            params,
                            timestepper, 
                            stage)

    p, n = @index(Global, NTuple)

    x⃗ⁱ = @inbounds positions[p][n, :]
    u⃗ⁱ = @inbounds velocities[p][n, :]

    if n == 1
        x⃗⁻ = zeros(3)
        u⃗ⁱ⁻¹ = zeros(3)
    else
        x⃗⁻ = @inbounds positions[p][n - 1, :]
        u⃗ⁱ⁻¹ = @inbounds velocities[p][n - 1, :]
    end

    Δx⃗ = x⃗ⁱ - x⃗⁻

    x, y, z = @inbounds [x_base[p], y_base[p], z_base[p]] + x⃗ⁱ

    l = sqrt(dot(Δx⃗, Δx⃗))
    Vᵖ = @inbounds pneumatocyst_volumes[p][n]

    Fᴮ = @inbounds (params.ρₒ - 500) * Vᵖ * [0.0, 0.0, params.g] #currently assuming kelp is nutrally buoyant except for pneumatocysts

    if @inbounds Fᴮ[3] > 0 && z >= 0  # i.e. floating up not sinking, and outside of the surface
        @inbounds Fᴮ[3] = 0.0
    end

    Aᵇ = @inbounds blade_areas[p][n]
    rˢ = @inbounds stipe_radii[p][n]
    Vᵐ = π * rˢ ^ 2 * l + Aᵇ * 0.01 # TODO: change thickness to some realistic thing
    mᵉ = (Vᵐ + params.Cᵃ * (Vᵐ + Vᵖ)) * params.ρₒ + Vᵖ * (params.ρₒ - 500) 

    u⃗ʷ = @inbounds interpolate.(values(water_velocities), x, y, z)
    u⃗ᵣₑₗ = u⃗ʷ - u⃗ⁱ
    sᵣₑₗ = sqrt(dot(u⃗ᵣₑₗ, u⃗ᵣₑₗ))

    a⃗ʷ = @inbounds interpolate.(values(water_accelerations), x, y, z)
    a⃗ⁱ = @inbounds accelerations[p][n, :]
    #a⃗ᵣₑₗ = a⃗ʷ - a⃗ⁱ 

    θ = acos(min(1, abs(dot(u⃗ᵣₑₗ, Δx⃗)) / (sᵣₑₗ * l + eps(0.0))))
    Aˢ = @inbounds 2 * rˢ * l * abs(sin(θ)) + π * rˢ * abs(cos(θ))

    Fᴰ = 0.5 * params.ρₒ * (params.Cᵈˢ * Aˢ + params.Cᵈᵇ * Aᵇ) * sᵣₑₗ .* u⃗ᵣₑₗ

    if n == @inbounds length(relaxed_lengths[p])
        x⃗⁺ = x⃗ⁱ 
        u⃗ⁱ⁺¹ = u⃗ⁱ
        Aᶜ⁺ = 0.0 
        l₀⁺ = @inbounds relaxed_lengths[p][n] 
    else
        x⃗⁺ = @inbounds positions[p][n + 1, :]
        u⃗ⁱ⁺¹ = @inbounds velocities[p][n + 1, :]
        Aᶜ⁺ = @inbounds π * stipe_radii[p][n + 1] ^ 2
        l₀⁺ = @inbounds relaxed_lengths[p][n + 1]
    end

    Aᶜ⁻ = @inbounds π * rˢ ^ 2
    l₀⁻ = @inbounds relaxed_lengths[p][n]

    Δx⃗⁻ = x⃗⁻ - x⃗ⁱ
    Δx⃗⁺ = x⃗⁺ - x⃗ⁱ

    #Δu⃗ⁱ⁻¹ = u⃗ⁱ⁻¹ - u⃗ⁱ
    #Δu⃗ⁱ⁺¹ = u⃗ⁱ⁺¹ - u⃗ⁱ

    l⁻ = sqrt(dot(Δx⃗⁻, Δx⃗⁻))
    l⁺ = sqrt(dot(Δx⃗⁺, Δx⃗⁺))

    T⁻ = tension(l⁻, l₀⁻, Aᶜ⁻, params) .* Δx⃗⁻ ./ (l⁻ + eps(0.0)) #+ ifelse(l⁻ > l₀⁻, params.kᵈ * Δu⃗ⁱ⁻¹, zeros(3))
    T⁺ = tension(l⁺, l₀⁺, Aᶜ⁺, params) .* Δx⃗⁺ ./ (l⁺ + eps(0.0)) #+ ifelse(l⁺ > l₀⁺, params.kᵈ * Δu⃗ⁱ⁺¹, zeros(3))

    Fⁱ = params. ρₒ * (Vᵐ + Vᵖ) .*  a⃗ʷ

    #if any(abs.(Fⁱ)/abs.(Fᴰ) .> 1000)
    #    @show Fⁱ
    #    @show Fᴰ
    #    Fⁱ *= maximum(abs, 1000 .* Fᴰ ./ Fⁱ)
    #end

    @inbounds begin 
        accelerations[p][n, :] .= (Fᴮ + Fᴰ + T⁻ + T⁺ + Fⁱ) ./ mᵉ - velocities[p][n, :] ./ params.τ
        drag_forces[p][n, :] .= Fᴰ + Fⁱ # store for back reaction onto water
        
        if any(isnan.(accelerations[p][n, :])) error("F is NaN: i=$i $(Fᴮ) .+ $(Fᴰ) .+ $(T⁻) .+ $(T⁺) at $x, $y, $z") end

        old_velocities[p][n, :] .= velocities[p][n, :]
        
        velocities[p][n, :] .+= timestepper(accelerations[p][n, :], old_accelerations[p][n, :], Δt, stage)
        
        old_accelerations[p][n, :] .= accelerations[p][n, :]

        positions[p][n, :] .+= timestepper(velocities[p][n, :], old_velocities[p][n, :], Δt, stage)

        if positions[p][n, 3] + z_base[p] > 0.0 #given above bouyancy conditions this should never be possible (assuming a flow with zero vertical velocity at the surface, i.e. a real one)
            positions[p][n, 3] = - z_base[p]
        end
    end
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

    step_kernel! = step_node!(device(model.architecture), workgroup, worksize)

    n_substeps = max(1, floor(Int, Δt / particles.parameters.max_Δt))

    water_accelerations = @inbounds model.timestepper.Gⁿ[(:u, :v, :w)]
    for substep in 1:n_substeps        
        for stage in stages(particles.parameters.timestepper)
            step_event = step_kernel!(particles.properties.x, 
                                      particles.properties.y, 
                                      particles.properties.z, 
                                      particles.properties.positions, 
                                      particles.properties.velocities, 
                                      particles.properties.pneumatocyst_volumes, 
                                      particles.properties.stipe_radii,  
                                      particles.properties.blade_areas, 
                                      particles.properties.relaxed_lengths, 
                                      particles.properties.accelerations, 
                                      particles.properties.drag_forces, 
                                      particles.properties.old_velocities, 
                                      particles.properties.old_accelerations, 
                                      model.velocities,
                                      water_accelerations, 
                                      Δt / n_substeps, 
                                      particles.parameters,
                                      particles.parameters.timestepper,
                                      stage)

            wait(step_event)
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
    x⃗_ = @inbounds transform.R₁ * (transform.R₂ * ([x, y, z] - transform.x⃗₀))

    if @inbounds x⃗_[3]>=0.0
        x_, y_, z_ = transform.R₃⁺ * x⃗_
    else
        x_, y_, z_ = transform.R₃⁻ * x⃗_
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

@kernel function weights!(drag_field, grid, rᵉ, l⁺, l⁻, polar_transform, parameters)
    i, j, k = @index(Global, NTuple)

    x, y, z = node(Center(), Center(), Center(), i, j, k, grid)

    x_, y_, z_ = polar_transform(x, y, z)
    r = (x_ ^ 2 + y_ ^ 2) ^ 0.5
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
                            scalefactors, positions, 
                            effective_radii, 
                            drag_forces, 
                            grid, drag_field, 
                            n_nodes, 
                            water_accelerations, 
                            weights_kernel!, 
                            apply_drag_kernel!, 
                            parameters)
                            
    p = @index(Global)
    scalefactor = @inbounds scalefactors[p]

    for n = 1:n_nodes
        # get node positions and size
        @inbounds begin
            x⃗ = @inbounds positions[p][n, :] + [base_x[p], base_y[p], base_z[p]]
            if n==1
                x⃗⁻ =  @inbounds [base_x[p], base_y[p], base_z[p]]
            else
            x⃗⁻ = @inbounds positions[p][n - 1, :] + [base_x[p], base_y[p], base_z[p]]
            end

            if n == n_nodes
                x⃗⁺ = x⃗
            else
                x⃗⁺ = @inbounds positions[p][n + 1, :] + [base_x[p], base_y[p], base_z[p]]
            end
        end

        rᵉ = @inbounds effective_radii[p][n]

        Δx⃗ = x⃗⁺ - x⃗⁻
                
        if n == 1
            l⁻ = @inbounds sqrt(dot(positions[p, n], positions[p, n]))
        else
            dp = @inbounds positions[p][n, :] - positions[p][n - 1, :]
            l⁻ = sqrt(dot(dp, dp))/2
        end
            
        θ = @inbounds atan(Δx⃗[2] / (Δx⃗[1] + eps(0.0))) + π * 0 ^ (1 + sign(Δx⃗[1]))
        ϕ = @inbounds atan((Δx⃗[1] ^ 2 + Δx⃗[2] ^ 2 + eps(0.0)) ^ 0.5 / Δx⃗[3])

        
        cosθ⁻ = dot(Δx⃗, x⃗ - x⃗⁻) / (sqrt(dot(Δx⃗, Δx⃗)) * sqrt(dot(x⃗ - x⃗⁻, x⃗ - x⃗⁻)))
        θ⁻ = -1.0 <= cosθ⁻ <= 1.0 ? acos(cosθ⁻) : 0.0

        if n == n_nodes
            dp = @inbounds positions[p][n, :] - positions[p][n - 1, :]
            l⁺ = sqrt(dot(dp, dp))/2
            θ⁺ = θ⁻
        else
            dp = @inbounds positions[p][n + 1, :] - positions[p][n, :]
            l⁺ = sqrt(dot(dp, dp))/2
            cosθ⁺ = - dot(Δx⃗, x⃗⁺ - x⃗) / (sqrt(dot(Δx⃗, Δx⃗)) * sqrt(dot(x⃗⁺ - x⃗, x⃗⁺ - x⃗)))
            θ⁺ = -1.0 <= cosθ⁺ <= 1.0 ? acos(cosθ⁺) : 0.0
        end

        weights_event = @inbounds weights_kernel!(drag_field[p], grid, rᵉ, l⁺, l⁻, LocalTransform(θ, ϕ, θ⁺, θ⁻, x⃗), parameters)
        wait(weights_event)

        normalisation = sum(@inbounds drag_field[p])
        Fᴰ = @inbounds drag_forces[p][n, :]

        # fallback if nodes are closer together than gridpoints and the line joining them is parallel to a grid Axis
        # as this means there are then no nodes in the stencil. This is mainly an issue for nodes close together lying on the surface
        # As long as the (relaxed) segment lengths are properly considered this shouldn't be an issue except during startup where upstream 
        # elements will quickly move towards dowmnstream elements
        @inbounds if normalisation == 0.0
            (ϵ, i), (η, j), (ζ, k) = modf.(fractional_indices(x⃗..., (Center(), Center(), Center()), grid))
            i, j, k = floor.(Int, (i, j, k))
            vol = Vᶜᶜᶜ(i, j, k, grid)
            inverse_effective_mass = 1 / (vol * parameters.ρₒ)
            water_accelerations.u[i, j, k] -= Fᴰ[1] * inverse_effective_mass * scalefactor
            water_accelerations.v[i, j, k] -= Fᴰ[2] * inverse_effective_mass * scalefactor
            water_accelerations.w[i, j, k] -= Fᴰ[3] * inverse_effective_mass * scalefactor

            @warn "Used fallback drag application as stencil found no nodes, this should be concerning if not in the initial transient response at $p, $n"
        else
            apply_drag_event = @inbounds apply_drag_kernel!(water_accelerations, drag_field[p], normalisation, grid, Fᴰ, scalefactor, parameters)
            wait(apply_drag_event)
        end
    end
end

function drag_water!(model)
    particles = model.particles
    water_accelerations = @inbounds model.timestepper.Gⁿ[(:u, :v, :w)]

    workgroup, worksize = work_layout(model.grid, :xyz)
    weights_kernel! = weights!(device(model.architecture), workgroup, worksize)
    apply_drag_kernel! = apply_drag!(device(model.architecture), workgroup, worksize)

    n_particles = length(particles)
    n_nodes = particles.parameters.n_nodes

    drag_water_kernel! = drag_node!(device(model.architecture), min(256, n_particles), n_particles)

    drag_nodes_event = drag_water_kernel!(particles.properties.x, particles.properties.y, particles.properties.z, 
                                          particles.properties.scalefactor, 
                                          particles.properties.positions, 
                                          particles.properties.effective_radii, 
                                          particles.properties.drag_forces, 
                                          model.grid, 
                                          particles.properties.drag_field, 
                                          n_nodes, 
                                          water_accelerations, 
                                          weights_kernel!, 
                                          apply_drag_kernel!, 
                                          particles.parameters)
    wait(drag_nodes_event)
end

end # module