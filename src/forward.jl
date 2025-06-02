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
Build the matrix of the linear system solved by the coefficients α and β
"""
function build_system(problem::ForwardProblem, j::Integer, σ::Vector)
    res = zeros(eltype(σ), 2*(problem.n-1), 2*(problem.n-1))

    for i=1:problem.n-1
        k = 2*i-1  # index corresponding to the i-th variable α
        # coefficient associated to α_i in the i-th first equation
        res[k, k] = problem.radii[i]^j
        # coefficient associated to β_i in the i-th first equation
        res[k, k+1] = 1/problem.radii[i]^j
        # coefficient associated to α_i in the i-th second equation
        res[k+1, k] = σ[i]*problem.radii[i]^j
        # coefficient associated to β_i in the i-th second equation
        res[k+1, k+1] = -σ[i]/problem.radii[i]^j
    end

    for i=1:problem.n-2
        k = 2*i-1  # index corresponding to the i-th variable α
        # coefficient associated to α_{i+1} in the i-th first equation
        res[k, k+2] = -problem.radii[i]^j
        # coefficient associated to β_{i+1} in the i-th first equation
        res[k, k+3] = -1/problem.radii[i]^j
        # coefficient associated to α_{i+1} in the i-th second equation
        res[k+1, k+2] = -σ[i+1]*problem.radii[i]^j  
        # coefficient associated to β_{i+1} in the i-th second equation
        res[k+1, k+3] = σ[i+1]/problem.radii[i]^j  
    end

    return res
end

"""
Solve for the coefficients α and β. Output contains (α1, β1, ..., α_{n-1}, β_{n-1})
"""
function solve_system(problem::ForwardProblem, j, σ)
    rhs = zeros(eltype(σ), 2*(problem.n-1))  # right hand side of the system
    k = 2*(problem.n-1)-1  # index corresponding to α_{n-1}

    # right hand side of the (n-1)-th first equation
    rhs[k] = problem.radii[problem.n-1]^j  
    # right hand side of the (n-1)-th second equation
    rhs[k+1] = σ[problem.n]*problem.radii[problem.n-1]^j

    L = build_system(problem, j, σ)

    # TODO: clean up
    # prob = LinearProblem(L, rhs)
    # sol = LinearSolve.solve(prob)
    # return sol.u

    return L\rhs
end

"""
Neumann-to-Dirichlet forward map
"""
function forward_map(problem::ForwardProblem, j, σ)
    αβ = solve_system(problem, j, σ)
    α1 = αβ[1]
    β1 = αβ[2]

    return (α1+β1)/(σ[1]*j*(α1-β1))
end