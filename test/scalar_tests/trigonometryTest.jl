@testset "Trigonometry Scalar" begin
    x_data = 2.0
    x = Variable(x_data, "x")
    
    z = @toposort sin(x)
    backward!(z)

    @test x.grad ≈ cos(x_data) atol=0.001

    x_data = 2.0
    x = Variable(x_data, "x")
    
    z = @toposort cos(x)
    backward!(z)

    @test x.grad ≈ -sin(x_data) atol=0.001

    x_data = 2.0
    x = Variable(x_data, "x")
    
    z = @toposort tan(x)
    backward!(z)

    @test x.grad ≈ 1/cos(x_data)^2 atol=0.001

    x_data = 2.0
    x = Variable(x_data, "x")
    
    z = @toposort cot(x)
    backward!(z)

    @test x.grad ≈ -1/sin(x.value)^2 atol=0.001

    x_data = 2.0
    x = Variable(x_data, "x")
    
    z = @toposort sec(x)
    backward!(z)

    @test x.grad ≈ sec(x_data)*tan(x_data) atol=0.001

    x_data = 2.0
    x = Variable(x_data, "x")
    
    z = @toposort csc(x)
    backward!(z)

    @test x.grad ≈ -cot(x.value)*csc(x.value) atol=0.001
end