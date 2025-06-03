"""
$(TYPEDEF)

Container for the convex nonlinear SDP problem

# Fields

$(TYPEDFIELDS)
"""
struct ConvexCalderonProblem
    "weight vector"
    c::Vector
    "conductivity lower bound"
    a::Number
    "conductivity upper bound"
    b::Number
    "observations"
    obs::Vector
    "forward problem"
    forward::ForwardProblem
end

"""
$(SIGNATURES)

Build the convex nonlinear SDP associated to the input Calderon problem
"""
function build_nonlinear_sdp(problem::ConvexCalderonProblem, σ_init::Vector; reg_param=0.0)
    obj(σ, p) = sum(problem.c .* σ)

    m = length(problem.obs)
    
    lb = problem.a .* ones(problem.forward.n)  # lower bound on variables
    ub = problem.b .* ones(problem.forward.n)  # upper bound on variables

    function cons(res, σ, p)
        for j=1:m
            res[j] = forward_map(problem.forward, j, σ)
        end
    end

    lcons = (-Inf).*ones(m)  # no lower bound constraint
    ucons = problem.obs .+ reg_param  # Harrach's SDP constraint

    optprob = OptimizationFunction(obj, Optimization.AutoForwardDiff(), cons=cons)
    prob = OptimizationProblem(optprob, σ_init, lcons=lcons, ucons=ucons, lb=lb, ub=ub)

    return prob
end

"""
$(SIGNATURES)

Build the feasibility problem used to estimate the weight vector
"""
function build_c_estimation_problem(σ::Matrix, a::Real, b::Real, m::Integer,        
                                    forward::ForwardProblem; max_last_coord=false)
    (n, nσ) = size(σ)
    jac_forward = zeros(m, n, nσ)

    Λ(σ) = [forward_map(forward, j, σ) for j=1:m]
    dΛ(σ) = jacobian(Λ, AutoForwardDiff(), σ)

    for l=1:nσ
        jac_forward[:, :, l] .= dΛ(σ[:, l])
    end

    model = Model()

    @variable(model, z[1:m, l=1:nσ] >= 0)
    @variable(model, c[1:n] >= 0)
    @variable(model, λ[1:n, 1:nσ] >= 0)
    @variable(model, μ[1:n, 1:nσ] >= 0)

    @constraint(model, c[1] == 1.0)

    @constraint(model, opt[i=1:n, l=1:nσ], 
                c[i] + μ[i, l] - λ[i, l] + sum(z[j, l] * jac_forward[j, i, l] for j=1:m) 
                == 0)

    # complementary slackness for box constraints
    @constraint(model, cs1[i=1:n, l=1:nσ], λ[i, l] * (σ[i, l] - a) == 0)
    @constraint(model, cs2[i=1:n, l=1:nσ], μ[i, l] * (σ[i, l] - b) == 0)

    if max_last_coord
        @objective(model, Max, c[n])
    end

    return model
end