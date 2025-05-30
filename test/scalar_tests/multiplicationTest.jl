# skonsultować

@testset "Multiplication Scalar" begin
    x_data = 2.0
    y_data = 3.0
    x = Variable(x_data, "x")
    y = Variable(y_data, "y")

    z = @toposort y * x * x
    backward!(z)
    
    @test x.grad == 12.0
    @test y.grad == 4.0
end