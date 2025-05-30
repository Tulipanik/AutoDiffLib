@testset "Multiplication Matrix" begin
    x_data = [1.0 2.0; 3.0 4.0]
    y_data = [5.0 6.0; 7.0 8.0]
    x = Variable(x_data, "x")
    y = Variable(y_data, "y")
    
    z = @toposort x * y * x
    backward!(z)

    @test x.grad == [119.0 139.0; 129.0 149.0]
    @test y.grad == [12.0 28.0; 18.0 42.0]

    x_data = [1.0 2.0; 3.0 4.0]
    y_data = [5.0 6.0; 7.0 8.0]
    x = Variable(x_data, "x")
    y = Variable(y_data, "y")
    
    z = @toposort x .* y .* x
    backward!(z)

    @test x.grad == [10.0 24.0; 42.0 64.0]
    @test y.grad == [1.0 4.0; 9.0 16.0]

    x_data = [2.0 4.0]
    y_data = [5.0 6.0; 7.0 8.0]
    x = Variable(x_data, "x")
    y = Variable(y_data, "y")
    
    z = @toposort x * y
    backward!(z)

    @test x.grad == [11.0 15.0]
    @test y.grad == [2.0 2.0; 4.0 4.0]

    x_data = [2.0 4.0]
    y_data = [5.0 6.0]
    x = Variable(x_data, "x")
    y = Variable(y_data, "y")
    
    z = @toposort x .* y
    backward!(z)

    @test x.grad == [5.0 6.0]
    @test y.grad == [2.0 4.0]

    # x_data = 2.0
    # y_data = [5.0; 6.0]
    # x = Variable(x_data, "x")
    # y = Variable(y_data, "y")
    
    # z = x .* y
    # backward(z)

    # @test x.grad == [11.0 15.0]
    # @test y.grad == [2.0 2.0; 4.0 4.0]
end
