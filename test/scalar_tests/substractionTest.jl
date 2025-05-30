@testset "Substraction Scalar" begin
    x_data = 2.0
    y_data = 3.0
    x = Variable(x_data, "x")
    y = Variable(y_data, "y")

    z = @toposort x - y - x
    backward!(z)
    
    @test x.grad == 0.0
    @test y.grad == -1.0
end