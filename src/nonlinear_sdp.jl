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

Estimate the weight vector defining the objective of the convex nonlinear SDP problem.
"""
function estimate_c(σ::Matrix, a::Real, b::Real, m::Integer, forward::ForwardProblem;
                    max_last_coord=false, quiet=true)
    (n, nσ) = size(σ)
    jac_forward = zeros(m, n, nσ)

    Λ(σ) = [forward_map(forward, j, σ) for j=1:m]
    dΛ(σ) = jacobian(Λ, AutoForwardDiff(), σ)

    for l=1:nσ
        jac_forward[:, :, l] .= dΛ(σ[:, l])
    end

    model = Model(Ipopt.Optimizer)
    if quiet
        set_attribute(model, "print_level", 0)  # set print level to 0
        set_attribute(model, "sb", "yes")  # remove Ipopt header
    end

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

    optimize!(model)

    return value.(c), model
end