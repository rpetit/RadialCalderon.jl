@testset "constructor" begin
    r = [0.5, 0.25]
    problem = ForwardProblem(r)

    @test problem.n == 3
end

@testset "forward map" begin
    r = [0.5, 0.25]
    problem = ForwardProblem(r)
    σ1 = 1.5
    σ2 = 0.5
    σ = [1.0, σ1, σ2]

    # test coherence with Harrach's computations
    c(j) = (1/σ1+1)*(σ1+σ2) + (1/σ1-1)*(σ1-σ2)*r[2]^(2*j)/r[1]^(2*j)
    d(j) = (1/σ1-1)*(σ1+σ2)*r[1]^(2*j) + (1/σ1+1)*(σ1-σ2)*r[2]^(2*j)

    @test forward_map(problem, 1, σ) ≈ (c(1)+d(1)) / (c(1)-d(1))
    @test forward_map(problem, 2, σ) ≈ (c(2)+d(2)) / (2*(c(2)-d(2)))
    @test forward_map(problem, 3, σ) ≈ (c(3)+d(3)) / (3*(c(3)-d(3)))

    # test equality with 1/(j*σ0) for a constant conductivity equal to σ0
    σ0 = 0.5 
    for j=1:3
        @test forward_map(problem, j, σ0.*ones(problem.n)) == 1/(j*σ0)
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
function solve_system(problem::ForwardProblem, j::Integer, σ::Vector)
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

@testset "build and solve system" begin
    r = [0.5, 0.25]
    n = length(r) + 1
    problem = ForwardProblem(r)
    σ = ones(n)

    L = build_system(problem, 1, σ)

    @test size(L) == (2*(n-1), 2*(n-1))

    αβ = solve_system(problem, 1, σ)

    @test length(αβ) == size(L, 1)

    # test coherence with Harrach's computations
    @test isapprox(αβ, [1.0, 0.0, 1.0, 0.0])
   
    # same test with σ different from ones(n) and j different from 1
    σ1 = 1.5
    σ2 = 0.5
    σ = [1.0, σ1, σ2]
    j = 2
    αβ = solve_system(problem, j, σ)

    α1 = 0.25 * ((1/σ1+1)*(σ1+σ2) + (1/σ1-1)*(σ1-σ2)*r[2]^(2*j)/r[1]^(2*j))
    @test αβ[1] ≈ α1
    β1 = 0.25 * ((1/σ1-1)*(σ1+σ2)*r[1]^(2*j) + (1/σ1+1)*(σ1-σ2)*r[2]^(2*j))
    @test αβ[2] ≈ β1
    α2 = 0.5 * (1+σ2/σ1)
    @test αβ[3] ≈ α2
    β2 = 0.5 * (1-σ2/σ1) * r[2]^(2*j)
    @test αβ[4] ≈ β2

    # test that introducing a fictious annulus does not modify results
    r = [0.5, 0.3, 0.25]
    problem = ForwardProblem(r)
    σ = [1.0, 1.5, 1.5, 0.5]
    j = 2
    αβ_bis = solve_system(problem, j, σ)

    @test all(αβ_bis[1:4] .≈ αβ[1:4])
    @test all(αβ_bis[5:6] .≈ αβ[3:4])
end

"""
Alternative implementation of the Neumann-to-Dirichlet forward map
"""
function forward_map_old(problem::ForwardProblem, j::Integer, σ::Vector)
    αβ = solve_system(problem, j, σ)
    α1 = αβ[1]
    β1 = αβ[2]

    return (α1+β1)/(σ[1]*j*(α1-β1))
end

@testset "consistency with old forward map" begin
    r = [0.5, 0.25]
    problem = ForwardProblem(r)
    σ = [1.2, 0.8, 0.9]

    for j=1:3
        @test forward_map_old(problem, j, σ) ≈ forward_map(problem, j, σ)
    end
end