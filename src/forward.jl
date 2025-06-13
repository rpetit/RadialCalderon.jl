"""
$(TYPEDEF)

Container for the forward Calderon problem

# Fields

$(TYPEDFIELDS)
"""
struct ForwardProblem
    "number of annuli"
    n::Integer
    "inner radii"
    radii::Vector

    function ForwardProblem(radii::Vector)
        n = length(radii) + 1
        return new(n, radii)
    end
end

"""
$(SIGNATURES)

Neumann-to-Dirichlet forward map
"""
function forward_map(problem::ForwardProblem, j::Integer, σ::Vector)
    if !(all(σ .> 0.0))
        throw(DomainError(σ, "Conductivity must be positive."))
    end

    C = 0.0

    for i=(problem.n-1):-1:1
        ρ = (σ[i]-σ[i+1]) / (σ[i]+σ[i+1])
        C = (ρ*problem.radii[i]^(2*j) + C) / (1 + ρ*C*problem.radii[i]^(-2*j))
    end

    return (1+C) / (j*σ[1]*(1-C))
end
