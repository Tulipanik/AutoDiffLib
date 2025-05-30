using AutoDiffLib
using Test

include("scalar_tests/additionTest.jl")
include("scalar_tests/substractionTest.jl")
include("scalar_tests/multiplicationTest.jl")
include("scalar_tests/divisionTest.jl")
include("scalar_tests/exponentioationTest.jl")
include("scalar_tests/logarithmTest.jl")
include("scalar_tests/trigonometryTest.jl")
include("scalar_tests/maxTest.jl")
include("scalar_tests/otherTest.jl")

include("./matrix_tests/additionTest.jl")
include("matrix_tests/substractionTest.jl")
include("matrix_tests/multiplicationTest.jl")
include("matrix_tests/divisionTest.jl")

@testset "Complicated Test" begin
    out_features = 10
    in_features = 2

    W = Variable(randn(out_features, in_features) * sqrt(2 / in_features), "W")
    b = Variable(zeros(out_features, 1), "b")

    x = Variable([1.0, 2.0], "x")

    z = Sigmoid(W * x + b)

    z2 = @toposort z

    backward!(z2)
end