"""
A coupled model for the motion (and in the future growth and biogeochemical interactions) of giant kelp (Macrocystis pyrifera).

Based on the models proposed by [Utter1996](@citet) and [Rosman2013](@citet), and used in [StrongWright2023](@citet).

Implemented in the framework of OceanBioME.jl[StrongWright2023](@citep) and the coupled with the fluid dynamics of Oceananigans.jl[Ramadhan2020](@citep).
"""
module GiantKelpDynamics

export GiantKelp, NothingBGC, RK3, Euler, UtterDenny

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
import Base: size, length, show, summary
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

    # forces on nodes and force history
        accelerations :: VF
       old_velocities :: VF
    old_accelerations :: VF
          drag_forces :: VF

    kinematics :: KP

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
                       accelerations::VF,
                       old_velocities::VF,
                       old_accelerations::VF,
                       drag_forces::VF,
                       kinematics::KP,
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
                                                       accelerations,
                                                       old_velocities,
                                                       old_accelerations,
                                                       drag_forces,
                                                       kinematics,
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

"""
    nothingfunc(args...)

Returns nothing for `nothing(args...)`
"""
@inline nothingfunc(args...) = nothing

"""
    GiantKelp(; grid, 
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
                kinematics = UtterDenny(),
                timestepper = Euler(),
                max_Δt = Inf,
                tracer_forcing = NamedTuple(),
                custom_dynamics = nothingfunc)

Constructs a model of giant kelps with bases at `holdfast_x`, `_y`, `_z`.


Keyword Arguments
=================

- `grid`: (required) the geometry to build the model on
- `holdfast_x`, `holdfast_y`, `holdfast_z`: An array of the base/holdfast positions of the individuals
- `scalefactor`: array of the scalefactor for each plant (used to allow each plant model to represnt the effect of multiple individuals)
- `number_nodes`: the number of nodes to split each individual interior
- `segment_unstretched_length`: either a scalar specifying the unstretched length of all segments, 
   or an array of the length of each segment (at the moment each plant must have the same)
- `initial_stipe_radii`: either a scalar specifying the stipe radii of all segments, 
   or an array of the stipe radii of each segment (at the moment each plant must have the same)
- `initial_blade_areas`: an array of the blade area attatched to each segment
- `initial_pneumatocyst_volume`: an array of the volume of pneumatocyst attatched to each segment
- `kinematics`: the kinematics model specifying the individuals motion
- `timestepper`: the timestepper to integrate the motion with (at each substep)
- `max_Δt`: the maximum timestep for integrating the motion
- `tracer_forcing`: a `NamedTuple` of `Oceananigans.Forcings(func; field_dependencies, parameters)` with functions 
  of the form `func(field_dependencies..., parameters)` where `field_dependencies` can be particle properties or 
  fields from the underlying model (tracers or velocities)
- `custom_dynamics`: function of the form `func(particles, model, bgc, Δt)` to be executed at every timestep after the kelp model properties are updated.

Example
=======

```jldoctest
julia> using GiantKelpDynamics, Oceananigans

julia> grid = RectilinearGrid(size=(16, 16, 16), extent=(100, 100, 8));

julia> kelp = GiantKelp(; grid, holdfast_x = [10., 20.], holdfast_y = [10., 20], holdfast_z = [-8., -8.])
Giant kelp (Macrocystis pyrifera) model with 2 individuals of 8 nodes. 
 Base positions:
 - x ∈ [10.0, 20.0]
 - y ∈ [10.0, 20.0]
 - z ∈ [-8.0, -8.0]

```
"""
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
                     kinematics = UtterDenny(),
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
                     accelerations,
                     old_velocities, old_accelerations,
                     drag_forces,
                     kinematics,
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
                                                 adapt(to, kelp.accelerations),
                                                 adapt(to, kelp.old_velocities),
                                                 adapt(to, kelp.old_accelerations),
                                                 adapt(to, kelp.drag_forces),
                                                 adapt(to, kelp.kinematics),
                                                 nothing,
                                                 adapt(to, kelp.max_Δt),
                                                 nothing,
                                                 nothing)

size(particles::GiantKelp) = size(particles.holdfast_x)
length(particles::GiantKelp) = length(particles.holdfast_x)

summary(particles::GiantKelp) = string("Giant kelp (Macrocystis pyrifera) model with $(length(particles)) individuals of $(size(particles.positions[1], 1)) nodes.")
show(io::IO, particles::GiantKelp) = print(io, string(summary(particles), " \n",
                                                      " Base positions:\n", 
                                                      " - x ∈ [$(minimum(particles.holdfast_x)), $(maximum(particles.holdfast_x))]\n",
                                                      " - y ∈ [$(minimum(particles.holdfast_y)), $(maximum(particles.holdfast_y))]\n",
                                                      " - z ∈ [$(minimum(particles.holdfast_z)), $(maximum(particles.holdfast_z))]"))

"""
    set!(kelp::GiantKelp; kwargs...)

Sets the properties of the `kelp` model. The keyword arguments kwargs... take the form name=data, where name refers to one of the properties of
`kelp`, and the data may be an array mathcing the size of the property for one individual (i.e. size(kelp.name[1])), or for all (i.e. size(kelp.name)).

Example
=======

```jldoctest
julia> using GiantKelpDynamics, Oceananigans

julia> grid = RectilinearGrid(size=(16, 16, 16), extent=(100, 100, 8));

julia> kelp = GiantKelp(; grid, number_nodes = 2, holdfast_x = [10., 20.], holdfast_y = [10., 20], holdfast_z = [-8., -8.])
Giant kelp (Macrocystis pyrifera) model with 2 individuals of 2 nodes. 
 Base positions:
 - x ∈ [10.0, 20.0]
 - y ∈ [10.0, 20.0]
 - z ∈ [-8.0, -8.0]

julia> set!(kelp, positions = [0 0 8; 8 0 8])

julia> set!(kelp, positions = [[0 0 8; 8 0 8], [0 0 -8; 8 0 -8]])

```
"""
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

"""
    NothingBGC()

An Oceananigans `AbstractContinuousFormBiogeochemistry` which specifies no biogeochemical
interactions to allow the giant kelp model to be run alone.

Example
=======

```jldoctest
julia> using GiantKelpDynamics, Oceananigans, OceanBioME

julia> grid = RectilinearGrid(size=(16, 16, 16), extent=(100, 100, 8));

julia> kelp = GiantKelp(; grid, number_nodes = 2, holdfast_x = [10., 20.], holdfast_y = [10., 20], holdfast_z = [-8., -8.])
Giant kelp (Macrocystis pyrifera) model with 2 individuals of 2 nodes. 
 Base positions:
 - x ∈ [10.0, 20.0]
 - y ∈ [10.0, 20.0]
 - z ∈ [-8.0, -8.0]

julia> biogeochemistry = Biogeochemistry(NothingBGC(); particles = kelp)
No biogeochemistry 
 Light attenuation: Nothing
 Sediment: Nothing
 Particles: Giant kelp (Macrocystis pyrifera) model with 2 individuals of 2 nodes.

```
"""
struct NothingBGC <: AbstractContinuousFormBiogeochemistry end

summary(::NothingBGC) = string("No biogeochemistry")
show(io, ::NothingBGC) = print(io, string("No biogeochemistry"))
show(::NothingBGC) = string("No biogeochemistry") # show be removed when show for `Biogeochemistry` is corrected

include("timesteppers.jl")
include("kinematics/Kinematics.jl")
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

            # I want to remove the water density thing here to be computed correctly, not not sure how to at the moment
            cell_mass = volume(i, j, k, model.grid, Center(), Center(), Center()) * vertical_spread * particles.kinematics.water_density

            apply_drag!(particles, Gᵘ, Gᵛ, Gʷ, i, j, k, k_base, cell_mass, p, n)

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
