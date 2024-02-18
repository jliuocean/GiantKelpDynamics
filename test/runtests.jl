using Oceananigans, GiantKelpDynamics, Test, Statistics, JLD2, Oceananigans.Units, CUDA
using OceanBioME: Biogeochemistry

arch = CUDA.has_cuda() ? GPU() : CPU()

include("kinematics.jl")
include("utils.jl")
include("tracer_release.jl")
