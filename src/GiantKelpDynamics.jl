module GiantKelpDynamics

export GiantKelp

using Adapt, KernelAbstractions

using KernelAbstractions.Extras: @unroll
using OceanBioME.Particles: BiogeochemicalParticles
using Oceananigans.Architectures: architecture, device
using Oceananigans.Fields: Field, CenterField, VelocityFields
using Oceananigans.Utils: arch_array

import Adapt: adapt_structure
import Base: size, length
import Oceananigans: set!
import Oceananigans.Biogeochemistry: update_tendencies!
import Oceananigans.Models.LagrangianParticleTracking: update_lagrangian_particle_properties!, _advect_particles!
import Oceananigans.OutputWriters: fetch_output, convert_output

struct GiantKelp{FT, VF, VI, SF, KP, TS, DT, DFS, CD} <: BiogeochemicalParticles
    # origin position
    holdfast_x :: FT
    holdfast_y :: FT
    holdfast_z :: FT

    scalefactor :: FT

    #information about nodes
    positions :: VF
    positions_ijk :: VI
    velocities :: VF 
    relaxed_lengths :: SF
    stipe_radii :: SF 
    blade_areas :: SF
    pneumatocyst_volumes :: SF 
    effective_radii :: SF # effective radius to drag over

    # forces on nodes and force history
    accelerations :: VF
    old_velocities :: VF
    old_accelerations :: VF
    drag_forces :: VF

    kinematic_parameters :: KP

    timestepper :: TS
    max_Δt :: DT

    drag_fields :: DFS

    custom_dynamics :: CD

    function GiantKelp(holdfast_x::FT, holdfast_y::FT, holdfast_z::FT,
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
                       kinematic_parameters::KP,
                       timestepper::TS,
                       max_Δt::DT,
                       drag_fields::DFS,
                       custom_dynamics::CD) where {FT, VF, VI, SF, KP, TS, DT, DFS, CD}

        return new{FT, VF, VI, SF, KP, TS, DT, DFS, CD}(holdfast_x, holdfast_y, holdfast_z,
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
                                                        kinematic_parameters,
                                                        timestepper,
                                                        max_Δt,
                                                        drag_fields,
                                                        custom_dynamics)
    end
end

function segment_area_fraction(lengths)
    fractional_length = cumsum(lengths) ./ sum(lengths)

    # Jackson et al. 1985 (https://www.jstor.org/stable/24817427)
    cumulative_areas = -0.08 .+ 3.3 .* fractional_length .- 4.1 .* fractional_length .^ 2 .+ 1.9 .* fractional_length .^ 3

    return cumulative_areas .- [0.0, cumulative_areas[1:end-1]...]
end

@inline nothingfunc(args...) = nothing

function GiantKelp(; grid, 
                     holdfast_x, holdfast_y, holdfast_z,
                     scalefactor = ones(length(holdfast_x)),
                     number_nodes = 8,
                     segment_unstretched_length = 0.6,
                     initial_stipe_radii = 0.004,
                     initial_blade_areas = 3.0 * (isa(segment_unstretched_length, Number) ? 
                                                    ones(number_nodes) ./ number_nodes :
                                                    segment_area_fraction(segment_unstretched_length)),
                     initial_pneumatocyst_volume = (2.5 / (5 * 9.81)) .* (isa(segment_unstretched_length, Number) ?
                                                                            1 / number_nodes .* ones(number_nodes) :
                                                                            segment_unstretched_length ./ sum(segment_unstretched_length)),
                     initial_effective_radii = 0.5 * ones(number_nodes), 
                     kinematic_parameters = (k = 1.91 * 10 ^ 7, 
                                             α = 1.41, 
                                             ρₒ = 1026.0, 
                                             ρₐ = 1.225, 
                                             g = 9.81, 
                                             Cᵈˢ = 1.0, 
                                             Cᵈᵇ= 0.4 * 12 ^ -0.485, 
                                             Cᵃ = 3.0,
                                             n_nodes = number_nodes,
                                             τ = 5.0,
                                             kᵈ = 500),
                     timestepper = RK3(),
                     max_Δt = Inf,
                     custom_dynamics = nothingfunc)

    number_kelp = length(holdfast_x)

    arch = architecture(grid)

    holdfast_x = arch_array(arch, holdfast_x)
    holdfast_y = arch_array(arch, holdfast_y)
    holdfast_z = arch_array(arch, holdfast_z)
    scalefactor = arch_array(arch, scalefactor)

    
    velocities = [arch_array(arch, zeros(number_nodes, 3)) for p in 1:number_kelp]
    positions = [arch_array(arch, zeros(number_nodes, 3)) for p in 1:number_kelp]

    positions_ijk = [arch_array(arch, ones(Int, number_nodes, 3)) for p in 1:number_kelp]


    relaxed_lengths = [arch_array(arch, ones(number_nodes) .* segment_unstretched_length) for p in 1:number_kelp]
    stipe_radii = [arch_array(arch, ones(number_nodes) .* initial_stipe_radii) for p in 1:number_kelp]
    blade_areas = [arch_array(arch, ones(number_nodes) .* initial_blade_areas) for p in 1:number_kelp]
    pneumatocyst_volumes = [arch_array(arch, ones(number_nodes) .* initial_pneumatocyst_volume) for p in 1:number_kelp]
    effective_radii = [arch_array(arch, ones(number_nodes) .* initial_effective_radii) for p in 1:number_kelp]

    accelerations = [arch_array(arch, zeros(number_nodes, 3)) for p in 1:number_kelp]
    old_velocities = [arch_array(arch, zeros(number_nodes, 3)) for p in 1:number_kelp]
    old_accelerations = [arch_array(arch, zeros(number_nodes, 3)) for p in 1:number_kelp]
    drag_forces = [arch_array(arch, zeros(number_nodes, 3)) for p in 1:number_kelp]

    drag_fields = VelocityFields(grid)

    return GiantKelp(holdfast_x, holdfast_y, holdfast_z,
                     scalefactor,
                     positions, positions_ijk,
                     velocities,
                     relaxed_lengths,
                     stipe_radii,
                     blade_areas,
                     pneumatocyst_volumes,
                     effective_radii,
                     accelerations,
                     old_velocities, old_accelerations,
                     drag_forces,
                     kinematic_parameters,
                     timestepper,
                     max_Δt,
                     drag_fields,
                     custom_dynamics)
end

adapt_structure(to, kelp::GiantKelp) = GiantKelp(adapt(to, kelp.holdfast_x),
                                                 adapt(to, kelp.holdfast_y), 
                                                 adapt(to, kelp.holdfast_z),
                                                 adapt(to, kelp.scalefactor),
                                                 adapt(to, kelp.positions),
                                                 adapt(to, kelp.positions_ijk),
                                                 adapt(to, kelp.velocities),
                                                 adapt(to, kelp.relaxed_lengths),
                                                 adapt(to, kelp.stipe_radii),
                                                 adapt(to, kelp.blade_areas),
                                                 adapt(to, kelp.pneumatocyst_volumes),
                                                 adapt(to, kelp.effective_radii),
                                                 adapt(to, kelp.accelerations),
                                                 adapt(to, kelp.old_velocities),
                                                 adapt(to, kelp.old_accelerations),
                                                 adapt(to, kelp.drag_forces),
                                                 adapt(to, kelp.kinematic_parameters),
                                                 nothing,
                                                 adapt(to, kelp.max_Δt),
                                                 adapt(to, kelp.drag_fields),
                                                 nothing)

size(particles::GiantKelp) = size(particles.holdfast_x)
length(particles::GiantKelp) = length(particles.holdfast_x)

function set!(kelp::GiantKelp; kwargs...)
    for (fldname, value) in kwargs
        ϕ = getproperty(kelp, fldname)
        if size(value) == size(ϕ) && size(value[1]) == size(ϕ[1])
            ϕ .= value
        elseif size(value) == size(ϕ[1])
            [ϕₚ .= value for ϕₚ in ϕ]
        else
            error("Size missmatch")
        end
    end
end

# for output writer

fetch_output(output::Union{Vector{<:Matrix}, Vector{<:Vector}}, model) = output

function convert_output(output::Union{Vector{<:Matrix}, Vector{<:Vector}}, writer)
    if architecture(output) isa GPU
        output_array = writer.array_type(undef, size(output)...)
        copyto!(output_array, output)
    else
        output_array = [convert(writer.array_type, opt) for opt in output]
    end

    return output_array
end

include("timesteppers.jl")
include("kinematics.jl")
include("drag_coupling.jl")

end # module GiantKelpDynamics
