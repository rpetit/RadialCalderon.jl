using RadialCalderon
using Test
using Aqua

@testset "Aqual.jl" begin
    Aqua.test_all(RadialCalderon)
end

@testset "Forward" begin
    include("test_forward.jl")
end