using Oceananigans, GiantKelpDynamics, Test, Statistics, JLD2
using OceanBioME: Biogeochemistry
using Oceananigans.Biogeochemistry: AbstractContinuousFormBiogeochemistry

struct NothingBGC <: AbstractContinuousFormBiogeochemistry end

include("kinematics.jl")
include("utils.jl")
