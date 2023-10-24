"""
A coupled model for the motion (and in the future growth and biogeochemical interactions) of giant kelp (Macrocystis pyrifera).

Based on the models proposed by [Utter1996](@citet) and [Rosman2013](@citet), and used in [StrongWright2023](@citet).

Implemented in the framework of OceanBioME.jl[StrongWright2023](@citep) and the coupled with the fluid dynamics of Oceananigans.jl[Ramadhan2020](@citep).
"""
module GiantKelpDynamics

export GiantKelp, NothingBGC, RK3, Euler

using Adapt, KernelAbstractions

using KernelAbstractions.Extras: @unroll
using OceanBioME.Particles: BiogeochemicalParticles
using Oceananigans: Center
using Oceananigans.Architectures: architecture, device
using Oceananigans.Biogeochemistry: AbstractContinuousFormBiogeochemistry
using Oceananigans.Fields: Field, CenterField, VelocityFields
using Oceananigans.Operators: volume
using Oceananigans.Utils: arch_array

import Adapt: adapt_structure
import Base: size, length
import Oceananigans: set!
import Oceananigans.Biogeochemistry: update_tendencies!
import Oceananigans.Models.LagrangianParticleTracking: update_lagrangian_particle_properties!, _advect_particles!
import Oceananigans.OutputWriters: fetch_output, convert_output

struct GiantKelp{FT, VF, VI, SF, KP, TS, DT, TF, CD} <: BiogeochemicalParticles
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

     tracer_forcing :: TF
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
                       tracer_forcing::TF,
                       custom_dynamics::CD) where {FT, VF, VI, SF, KP, TS, DT, TF, CD}

        return new{FT, VF, VI, SF, KP, TS, DT, TF, CD}(holdfast_x, holdfast_y, holdfast_z,
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
                                                       tracer_forcing,
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
                     segment_unstretched_length = 3.,
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
                     timestepper = Euler(),
                     max_Δt = Inf,
                     tracer_forcing = NamedTuple(),
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
                     tracer_forcing,
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
                                                 nothing,
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

# so that we can run with just kelp
struct NothingBGC <: AbstractContinuousFormBiogeochemistry end

include("timesteppers.jl")
include("kinematics.jl")
include("drag_coupling.jl")
include("forcing.jl")

function update_tendencies!(bgc, particles::GiantKelp, model)
    Gᵘ, Gᵛ, Gʷ = @inbounds model.timestepper.Gⁿ[(:u, :v, :w)]

    tracer_tendencies = @inbounds model.timestepper.Gⁿ[keys(particles.tracer_forcing)]

    n_nodes = @inbounds size(particles.positions[1], 1)

    #####
    ##### Apply the tracer tendencies from each particle
    ####
    # we have todo this serially for each particle otherwise we get unsafe memory access to the tendency field
    @inbounds @unroll for p in 1:length(particles)
        k_base = 1

        sf = particles.scalefactor[p]

        @unroll for n in 1:n_nodes
            i, j, k = particles.positions_ijk[p][n, :]
            vertical_spread = max(1, k - k_base  + 1)

            vol = volume(i, j, k, model.grid, Center(), Center(), Center()) * vertical_spread 

            apply_drag!(particles, Gᵘ, Gᵛ, Gʷ, i, j, k, k_base, vol, p, n)

            @unroll for kidx in k_base:k
                total_scaling = sf / vertical_spread 

                for (tracer_name, forcing) in pairs(particles.tracer_forcing)

                    tracer_tendency = tracer_tendencies[tracer_name]

                    forcing_arguments = get_arguments(forcing, particles, p, n)

                    forcing_tracers = @inbounds [model.tracers[tracer_name][i, j, kidx] for tracer_name in forcing.field_dependencies if tracer_name in keys(model.tracers)]

                    tracer_tendency[i, j, kidx] += total_scaling * forcing.func(forcing_tracers..., forcing_arguments..., forcing.parameters)
                end
            end

            k_base = k # maybe this should be k + 1
        end
    end
end



end # module GiantKelpDynamics
