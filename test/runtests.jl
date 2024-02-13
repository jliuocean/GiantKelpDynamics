using Oceananigans, GiantKelpDynamics, Test, Statistics, JLD2, Oceananigans.Units
using OceanBioME: Biogeochemistry

arch = CPU()

include("kinematics.jl")
include("utils.jl")
include("tracer_release.jl")
