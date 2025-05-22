"""
    RadialCalderon

A Julia package for HMM modeling, simulation, inference and learning.

# Exports

$(EXPORTS)
"""
module RadialCalderon

using DocStringExtensions

export ForwardProblem

include("forward.jl")

end
