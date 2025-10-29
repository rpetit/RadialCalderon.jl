# # Ill-posedness of the inverse problem

using RadialCalderon
using NonlinearSolve
using Random

Random.seed!(1234);  # fix the random seed for reproducibility

# ## Mean error for each annulus

n = 10
a = 0.5
b = 1.5
m = 10  # number of scalar measurements
nσ_true = 100  # number of unknown conductitivities
ϵ = 1e-15  # tolerance

σ_true_tab = a .+ (b-a) .* rand(n, nσ_true)

σ_hat_tab = zeros(n, nσ_true)
ninit_tab = zeros(Int, nσ_true)

r = reverse([i/n for i=1:n-1])
forward = ForwardProblem(r)

Λ(σ) = [forward_map(forward, j, σ) for j=1:m]

for iσ_true=1:nσ_true
    σ_true = σ_true_tab[:, iσ_true]
    obs_true = Λ(σ_true)

    converged = false
    ninit = 0
    σ_hat = zeros(n)

    while !converged
        ninit_tab[iσ_true] += 1

        σ_init = a .+ (b-a) .* rand(n)

        f(σ, p) = Λ(σ) .- obs_true
        problem = NonlinearProblem(f, σ_init)

        try
            res = NonlinearSolve.solve(problem, TrustRegion(), abstol=ϵ)
            σ_hat .= res.u
            converged = maximum(abs.(Λ(σ_hat) .- obs_true)) < ϵ
        catch error
            if !(error isa DomainError)
                rethrow(error)  # rethrow any other error
            end
        end
    end

    σ_hat_tab[:, iσ_true] .= σ_hat
end

#=
Now, we display the mean error on the $i$-th annulus as a function of $i$. The error on the 
outermost annulus is several orders of magnitude smaller than the error on the innermost annulus.
=#

using Statistics
using Plots
using LaTeXStrings

mean_err = vcat(mean(abs.(σ_true_tab .- σ_hat_tab), dims=2)...)

plot(mean_err, lc=:red, lw=2, linestyle=:dash, primary=false)
plot!(
    mean_err, 
    yscale=:log10, 
    ylim=(1e-14, 1e-7), 
    seriestype=:scatter, 
    ms=5,
    markerstrokewidth=2, 
    xticks=collect(1:10), 
    xtickfontsize=14, 
    ytickfontsize=14,
    primary=false, 
    formatter=:latex, 
    mc=:red, 
    linestyle=:dot
)
xlabel!(L"i", xguidefontsize=18)
ylabel!(L"\mathrm{mean}(|\hat{\sigma}_i-\sigma^\dagger_i|)", yguidefontsize=18)

# ## Mean error as a function of $n$

mean_err_tab = zeros(10)
mean_err_tab[10] = mean(maximum(abs.(σ_hat_tab .- σ_true_tab), dims=1)) 

for n=1:9
    global m = n

    σ_true_tab_n = a .+ (b-a) .* rand(n, nσ_true)    
    σ_hat_tab_n = zeros(n, nσ_true)

    global r = reverse([i/n for i=1:n-1])
    global forward = ForwardProblem(r)

    Λ(σ) = [forward_map(forward, j, σ) for j=1:m]

    for iσ_true=1:nσ_true
        σ_true = σ_true_tab_n[:, iσ_true]
        obs_true = Λ(σ_true)

        converged = false
        σ_hat = zeros(n)

        while !converged
            σ_init = a .+ (b-a) .* rand(n)

            f(σ, p) = Λ(σ) .- obs_true
            problem = NonlinearProblem(f, σ_init)

            try 
                res = NonlinearSolve.solve(problem, TrustRegion(), abstol=1e-15)
                σ_hat .= res.u
                converged = maximum(abs.(Λ(σ_hat) .- obs_true)) < 1e-15
            catch error
                if !(error isa DomainError)
                    rethrow(error)  # rethrow any other error
                end
            end
        end

        σ_hat_tab_n[:, iσ_true] .= σ_hat
    end
 
    mean_err_tab[n] = mean(maximum(abs.(σ_hat_tab_n .- σ_true_tab_n), dims=1))
end

#=
Now, we display the mean error as a function of $n$. The error significantly increases as n increases.
=#

plot(1:10, mean_err_tab, lc=:red, lw=2, linestyle=:dash, primary=false)
plot!(
    1:10,
    mean_err_tab,
    yscale=:log10, 
    ylim=(1e-17, 1e-7), 
    seriestype=:scatter, 
    ms=5,
    markerstrokewidth=2,
    xticks=collect(1:10), 
    yticks=1 ./ (10) .^ reverse(collect(7:17)),
    xtickfontsize=14, 
    ytickfontsize=14, 
    primary=false, 
    formatter=:latex, 
    mc=:red,
    linestyle=:dot
)
xlabel!(L"n", xguidefontsize=18)
ylabel!(L"\mathrm{mean}(\Vert\hat{\sigma}-\sigma^\dagger\Vert_{\infty})", yguidefontsize=18)