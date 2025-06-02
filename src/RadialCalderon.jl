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
using Ipopt

export ForwardProblem, forward_map
export ConvexCalderonProblem, estimate_c

include("forward.jl")
include("nonlinear_sdp.jl")

end
