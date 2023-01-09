module GiantKelpDynamics

export GiantKelp, kelp_dynamics!, fully_resolved_drag!, DiscreteDrag, DiscreteDragSet

using KernelAbstractions, LinearAlgebra, StructArrays
using KernelAbstractions.Extras: @unroll
using Oceananigans.Architectures: device, arch_array
using Oceananigans.Fields: _interpolate, fractional_indices
using Oceananigans.Utils: work_layout
using Oceananigans.Operators: Vᶜᶜᶜ
using Oceananigans: CPU, node, Center, CenterField
using Oceananigans.LagrangianParticleTracking: AbstractParticle, LagrangianParticles
using Oceananigans.Grids: node

import Adapt: adapt_structure

include("timesteppers.jl")

x⃗₀(number, depth, l₀::Number, initial_stretch::Number) = x⃗₀(number, depth, repeat([l₀], number), repeat([initial_stretch], number))
x⃗₀(number, depth, l₀::Array, initial_stretch::Number) = x⃗₀(number, depth, l₀, repeat([initial_stretch], number))
x⃗₀(number, depth, l₀::Number, initial_stretch::Array) = x⃗₀(number, depth, repeat([l₀], number), initial_stretch)

function x⃗₀(number, depth, l₀::Array, initial_stretch::Array)
    x = zeros(number, 3)
    for i in 1:number
        if sum(l₀[1:i]) * initial_stretch[i] - depth < 0
            x[i, 3] = sum(l₀[1:i] .* initial_stretch[1:i])
        else
            x[i, :] = [sum(l₀[1:i] .* initial_stretch[1:i]) - depth, 0.0, depth]
        end
    end
    return x
end

struct GiantKelp{FT, VF, VI, SF, FA} <: AbstractParticle
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
    positions_ijk :: VI
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

    function GiantKelp(x::FT, y::FT, z::FT,
                       fixed_x::FT, fixed_y::FT, fixed_z::FT,
                       scalefactor::FT,
                       positions::VF,
                       positions_ijk::VI,
                       velocities::VF,
                       relaxed_lengths::SF,
                       stipe_radii::SF,
                       blade_areas::SF,
                       pneumatocyst_volumes::SF,
                       effective_radii::SF,
                       accelerations::VF,
                       old_velocities::VF,
                       old_accelerations::VF,
                       drag_forces::VF,
                       drag_field::FA) where {FT, VF, VI, SF, FA}
        return new{FT, VF, VI, SF, FA}(x, y, z, fixed_x, fixed_y, fixed_z,
                                       scalefactor,
                                       positions,
                                       positions_ijk,
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
                                       drag_field)
    end
end

adapt_structure(to, kelp::GiantKelp) = GiantKelp(kelp.x, kelp.y, kelp.z,
                                                 kelp.fixed_x, kelp.fixed_y, kelp.fixed_z,
                                                 kelp.scalefactor,
                                                 adapt_structure(to, kelp.positions),
                                                 adapt_structure(to, kelp.positions_ijk),
                                                 adapt_structure(to, kelp.velocities),
                                                 adapt_structure(to, kelp.relaxed_lengths),
                                                 adapt_structure(to, kelp.stipe_radii),
                                                 adapt_structure(to, kelp.blade_areas),
                                                 adapt_structure(to, kelp.pneumatocyst_volumes),
                                                 adapt_structure(to, kelp.effective_radii),
                                                 adapt_structure(to, kelp.accelerations),
                                                 adapt_structure(to, kelp.old_velocities),
                                                 adapt_structure(to, kelp.old_accelerations),
                                                 adapt_structure(to, kelp.drag_forces),
                                                 adapt_structure(to, kelp.drag_field))

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
                      initial_node_positions = nothing,
                      initial_stretch = 2.0,
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
                      drag_fields = true,
                      max_Δt = Inf) where {FT}

    base_x = arch_array(architecture, base_x)
    base_y = arch_array(architecture, base_y)
    base_z = arch_array(architecture, base_z)
    scalefactor = arch_array(architecture, scalefactor)

    AFT = typeof(base_x)
    
    positions = []
    velocities = [arch_array(architecture, zeros(number_nodes, 3)) for p in 1:number_kelp]

    if isnothing(initial_node_positions)
        for i in 1:number_kelp
            push!(positions, arch_array(architecture, x⃗₀(number_nodes, depth, segment_unstretched_length, initial_stretch)))
        end
    else
        if !(size(initial_node_positions) == (number_nodes, 3))
            error("initial_node_positions is the wrong shape (should be ($number_nodes, 3))")
        end

        for i in 1:number_kelp
            push!(positions, arch_array(architecture, copy(initial_node_positions)))
        end
    end

    VF = typeof(positions) # float vector array type

    positions_ijk = []

    for i in 1:number_kelp
        push!(positions_ijk, arch_array(architecture, zeros(Int, number_nodes, 3)))
    end

    VI = typeof(positions_ijk)

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

    drag_field = drag_fields ? [CenterField(grid) for p in 1:number_kelp] : [nothing for p in 1:number_kelp]

    FA = typeof(drag_field)

    kelps = StructArray{GiantKelp{AFT, VF, VI, SF, FA}}((base_x, base_y, base_z, 
                                                         copy(base_x), copy(base_y), copy(base_z), 
                                                         scalefactor, 
                                                         positions, 
                                                         positions_ijk,
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

include("dynamics.jl")
include("full_drag.jl")
include("fast_drag.jl")
end # module