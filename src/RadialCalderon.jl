"""
    RadialCalderon

A Julia package for studying the Calderon problem with piecewise constant radial conductivities.

# Exports

$(EXPORTS)
"""
module RadialCalderon

using DocStringExtensions
using DifferentiationInterface
using ForwardDiff
using JuMP

export ForwardProblem, forward_map
export ConvexCalderonProblem, build_c_estimation_problem

include("forward.jl")
include("nonlinear_sdp.jl")

end
