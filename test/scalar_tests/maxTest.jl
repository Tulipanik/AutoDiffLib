@testset "Max Scalar" begin
    x_data = 2
    y_data = 3
    x = Variable(x_data, "x")
    y = Variable(y_data, "y")
    
    z = @toposort max(x, y)
    backward!(z)

    @test x.grad == 0
    @test y.grad == 1

    x_data = 3
    y_data = 2
    x = Variable(x_data, "x")
    y = Variable(y_data, "y")
    
    z = @toposort max(x, y)
    backward!(z)

    @test x.grad == 1
    @test y.grad == 0

    x_data = 2.0
    y_data = 2.0
    x = Variable(x_data, "x")
    y = Variable(y_data, "y")
    
    z = @toposort max(x, y)
    backward!(z)

    @test x.grad == 0.5
    @test y.grad == 0.5
end