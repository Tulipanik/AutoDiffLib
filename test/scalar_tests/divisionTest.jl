# skonsultować

@testset "Division Scalar" begin
    x_data = 2.0
    y_data = 3.0
    x = Variable(x_data, "x")
    y = Variable(y_data, "y")

    z = y / x
    backward(z)
    
    @test x.grad == -0.75
    @test y.grad == 0.5
end