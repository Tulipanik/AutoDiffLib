import Base: +, -, *, /, ^, sin, cos, tan, cot, sec, csc, exp, log, max
using Statistics

promote_to_node(x::Node) = x
promote_to_node(x::Number) = Constant(x)
promote_to_node(x::AbstractArray) = Constant(x)

function +(x::Node{T}, y::Node{T}) where {T<:Number}
    data = x.value + y.value
    inputs = [x, y]
    return Operation(inputs, data, z_grad::T -> [z_grad, z_grad], (x, y) -> x.value + y.value, "+")
end

+(x::Number, y::Node{T}) where {T<:Number} = promote_to_node(x) + y
+(x::Node{T}, y::Number) where {T<:Number} = x + promote_to_node(y)

function -(x::Node{T}, y::Node{T}) where {T<:Number}
    data = x.value - y.value
    inputs = [x, y]
    return Operation(inputs, data, z_grad::T -> [z_grad, -z_grad], (x, y) -> x.value - y.value, "-")
end

-(x::Number, y::Node{T}) where {T<:Number} = promote_to_node(x) - y
-(x::Node{T}, y::Number) where {T<:Number} = x - promote_to_node(y)

function *(x::Node{T}, y::Node{T}) where {T<:Number}
    data = x.value * y.value
    inputs = [x, y]
    return Operation(inputs, data, z_grad::T -> [y.value * z_grad, x.value * z_grad], (x, y) -> x.value * y.value, "*")
end

*(x::Number, y::Node{T}) where {T<:Number} = promote_to_node(x) * y
*(x::Node{T}, y::Number) where {T<:Number} = x * promote_to_node(y)

function /(x::Node{T}, y::Node{T}) where {T<:Number}
    data = x.value / y.value
    inputs = [x, y]
    return Operation(inputs, data, z_grad::T -> [z_grad / y.value, -(x.value * z_grad) / y.value^2], (x, y) -> x.value / y.value, "/")
end

/(x::Number, y::Node{T}) where {T<:Number} = promote_to_node(x) / y
/(x::Node{T}, y::Number) where {T<:Number} = x / promote_to_node(y)

function ^(x::Node{T}, y::Node{T}) where {T<:Number}
    data = x.value^y.value
    inputs = [x, y]
    return Operation(inputs, data, z_grad::T -> [y.value * x.value^(y.value - 1) * z_grad, log(x.value) * x.value^y.value * z_grad], (x, y) -> x.value^y.value, "^")
end

^(x::Number, y::Node{T}) where {T<:Number} = promote_to_node(x)^y
^(x::Node{T}, y::Number) where {T<:Number} = x^promote_to_node(y)

function exp(x::Node{T}) where {T<:Number}
    data = exp(x.value)
    inputs = [x]
    return Operation(inputs, data, z_grad::T -> [z_grad * exp(x.value)], (x) -> exp(x.value), "exp")
end

function log(x::Node{T}) where {T<:Number}
    data = log(x.value)
    inputs = [x]
    return Operation(inputs, data, z_grad::T -> [z_grad * (1 / x.value)], (x) -> log(x.value), "log")
end

function sin(x::Node{T}) where {T<:Number}
    data = sin(x.value)
    inputs = [x]
    return Operation(inputs, data, z_grad::T -> [z_grad * cos(x.value)], (x) -> sin(x.value), "sin")
end

function cos(x::Node{T}) where {T<:Number}
    data = cos(x.value)
    inputs = [x]
    return Operation(inputs, data, z_grad::T -> [z_grad * -sin(x.value)], (x) -> cos(x.value), "cos")
end

function tan(x::Node{T}) where {T<:Number}
    data = tan(x.value)
    inputs = [x]
    return Operation(inputs, data, z_grad::T -> [z_grad / cos(x.value)^2], (x) -> tan(x.value), "tan")
end

function cot(x::Node{T}) where {T<:Number}
    data = cot(x.value)
    inputs = [x]
    return Operation(inputs, data, z_grad::T -> [-z_grad / sin(x.value)^2], (x) -> cot(x.value), "cot")
end

function sec(x::Node{T}) where {T<:Number}
    data = sec(x.value)
    inputs = [x]
    return Operation(inputs, data, z_grad::T -> [z_grad * sec(x.value) * tan(x.value)], (x) -> sec(x.value), "sec")
end

function csc(x::Node{T}) where {T<:Number}
    data = csc(x.value)
    inputs = [x]
    return Operation(inputs, data, z_grad::T -> [-z_grad * cot(x.value) * csc(x.value)], (x) -> csc(x.value), "csc")
end

function max(x::Node{T}, y::Node{T}) where {T<:Number}
    data = max(x.value, y.value)
    inputs = [x, y]
    if x.value > y.value
        return Operation(inputs, data, z_grad::T -> [z_grad, zero(T)], (x, y) -> max(x.value, y.value), "max")
    elseif y.value > x.value
        return Operation(inputs, data, z_grad::T -> [zero(T), z_grad], (x, y) -> max(x.value, y.value), "max")
    else
        return Operation(inputs, data, z_grad::T -> [z_grad / 2, z_grad / 2], (x, y) -> max(x.value, y.value), "max")
    end
end

max(x::Number, y::Node{T}) where {T<:Number} = promote_to_node(x) |> (nx -> max(nx, y))
max(x::Node{T}, y::Number) where {T<:Number} = promote_to_node(y) |> (ny -> max(x, ny))

function Sigmoid(x::Node{T}) where {T<:Number}
    data = Sigmoid(x.value)
    inputs = [x]
    return Operation(inputs, data, z_grad::T -> [z_grad * data * (1 - data)], (x) -> Sigmoid(x.value), "Sigmoid")
end

function ReLU(x::Node{T}) where {T<:Number}
    data = ReLU(x.value)
    inputs = [x]

    if x.value > 0
        return Operation(inputs, data, z_grad::T -> [z_grad], (x) -> ReLU(x.value), "ReLU")
    else
        return Operation(inputs, data, z_grad::T -> [zero(T)], (x) -> ReLU(x.value), "ReLU")
    end
end

function +(x::Node{<:AbstractArray}, y::Node{<:AbstractArray})
    x_val, y_val = x.value, y.value

    foward = (a, b) -> begin
        a_sz = size(a)
        b_sz = size(b)
        if a_sz == b_sz
            return a .+ b
        elseif length(a_sz) == 2 && length(b_sz) == 1
            if b_sz[1] == a_sz[1]
                return a .+ reshape(b, :, 1)
            elseif b_sz[1] == a_sz[2]
                return a .+ reshape(b, 1, :)
            else
                throw(DimensionMismatch("Vector length does not match any matrix dimension"))
            end
        elseif length(a_sz) == 1 && length(b_sz) == 2
            if a_sz[1] == b_sz[1]
                return reshape(a, :, 1) .+ b
            elseif a_sz[1] == b_sz[2]
                return reshape(a, 1, :) .+ b
            else
                throw(DimensionMismatch("Vector length does not match any matrix dimension"))
            end
        else
            throw(DimensionMismatch("Unsupported shapes for + operation"))
        end
    end

    data = foward(x_val, y_val)
    inputs = [x, y]

    function backward(z_grad)
        x_shape = size(x.value)
        y_shape = size(y.value)

        gx = reshape(sum(z_grad, dims=setdiff(1:ndims(z_grad), findall(x_shape .!= 1))), x_shape)
        gy = reshape(sum(z_grad, dims=setdiff(1:ndims(z_grad), findall(y_shape .!= 1))), y_shape)
        return [gx, gy]
    end
    return Operation(inputs, data, backward, foward, "+")
end

function +(x::Node{<:Number}, y::Node{<:AbstractArray})
    data = fill(x.value, size(y.value)) .+ y.value
    inputs = [x, y]
    return Operation(inputs, data, z_grad -> [z_grad, z_grad], (x, y) -> fill(x.value, size(y.value)) .+ y.value, "+")
end


+(x::AbstractArray, y::Node{T}) where {T<:AbstractArray} = promote_to_node(x) + y
+(x::Node{T}, y::AbstractArray) where {T<:AbstractArray} = x + promote_to_node(y)

function -(x::Node{T}, y::Node{T}) where {T<:AbstractArray}
    data = x.value .- y.value
    inputs = [x, y]
    return Operation(inputs, data, z_grad::T -> [z_grad, -z_grad], (x, y) -> x.value .- y.value, "-")
end

-(x::AbstractArray, y::Node{T}) where {T<:AbstractArray} = promote_to_node(x)
-(x::Node{T}, y::AbstractArray) where {T<:AbstractArray} = promote_to_node(y)


function *(x::Node{<:AbstractArray}, y::Node{<:AbstractArray})
    data = x.value * y.value
    inputs = [x, y]
    return Operation(inputs, data, z_grad::AbstractArray -> [z_grad * y.value', x.value' * z_grad], (x, y) -> x.value * y.value, "*")
end

*(x::AbstractArray, y::Node{T}) where {T<:AbstractArray} = promote_to_node(x) * y
*(x::Node{T}, y::AbstractArray) where {T<:AbstractArray} = x * promote_to_node(y)


function Base.Broadcast.broadcasted(::typeof(*), x::Node{<:AbstractArray}, y::Node{<:AbstractArray})
    data = x.value .* y.value
    inputs = [x, y]
    return Operation(inputs, data, z_grad::AbstractArray -> [z_grad .* y.value, z_grad .* x.value], (x, y) -> x.value .* y.value, ".*")
end

function Base.Broadcast.broadcasted(::typeof(*), x::Node{<:Number}, y::Node{<:AbstractArray})
    data = x.value .* y.value
    inputs = [x, y]
    return Operation(inputs, data, z_grad -> [sum(z_grad .* y.value), z_grad .* x.value], (x, y) -> x.value .* y.value, ".*")
end

function Base.Broadcast.broadcasted(::typeof(*), x::Node{<:AbstractArray}, y::Node{<:Number})
    data = x.value .* y.value
    inputs = Node[x, y]
    return Operation(inputs, data, z_grad -> [z_grad .* y.value, sum(z_grad .* x.value)], (x, y) -> x.value .* y.value, ".*")
end

function /(x::Node{<:AbstractArray}, y::Node{<:AbstractArray})
    data = x.value / y.value
    inputs = [x, y]
    Yinv = inv(y.value)
    return Operation(inputs, data, z_grad::AbstractArray -> [z_grad * Yinv', -x.value' * z_grad * Yinv'], (x, y) -> x.value / y.value, "/")
end

/(x::AbstractArray, y::Node{T}) where {T<:AbstractArray} = promote_to_node(x) / y
/(x::Node{T}, y::AbstractArray) where {T<:AbstractArray} = x / promote_to_node(y)


function Base.Broadcast.broadcasted(::typeof(/), x::Node{<:AbstractArray}, y::Node{<:AbstractArray})
    data = x.value ./ y.value
    inputs = [x, y]
    return Operation(inputs, data, z_grad -> [z_grad ./ y.value, -z_grad .* x.value ./ (y.value .^ 2)], (x, y) -> x.value ./ y.value, "./")
end

function Base.Broadcast.broadcasted(::typeof(/), x::Node{<:AbstractArray}, y::Node{<:Number})
    data = x.value ./ y.value
    inputs = [x, y]
    return Operation(inputs, data, z_grad -> [z_grad ./ y.value, -sum(z_grad .* x.value) / y.value^2], (x, y) -> x.value ./ y.value, "./")
end

function Base.Broadcast.broadcasted(::typeof(/), x::Node{<:Number}, y::Node{<:AbstractArray})
    data = x.value ./ y.value
    inputs = [x, y]
    return Operation(inputs, data, z_grad -> [sum(z_grad ./ y.value), -z_grad .* x.value ./ (y.value .^ 2)], (x, y) -> x.value ./ y.value, "./")
end

function Base.Broadcast.broadcasted(::typeof(^), x::Node{<:AbstractArray}, y::Node{<:Number})
    data = x.value .^ y.value
    inputs = [x, y]
    return Operation(inputs, data, z_grad::AbstractArray -> [
            z_grad .* (y.value .* (x.value .^ (y.value - 1))),
            sum(z_grad .* (data .* log.(max.(x.value, eps()))))
        ], (x, y) -> x.value .^ y.value, ".^")
end

function log(x::Node{<:AbstractArray})
    data = log.(x.value)
    inputs = [x]
    return Operation(inputs, data, z_grad::AbstractArray -> z_grad .* (1 ./ x.value), (x) -> log.(x.value), ".log")
end

function mean(x::Node{<:AbstractArray})
    data = Statistics.mean(x.value)
    n = length(x.value)
    inputs = [x]

    backward = z_grad -> fill(z_grad / n, size(x.value))
    foward = x -> Statistics.mean(x)

    return SpreadedOperator(inputs, data, backward, foward, "mean")
end

function ReLU(x::Node{<:AbstractArray})
    data = ReLU.(x.value)
    inputs = [x]
    return Operation(inputs, data, z_grad::AbstractArray -> z_grad .* (x.value .>= 0), (x) -> ReLU.(x.value), ".ReLU")
end

function Sigmoid(x::Node{<:AbstractArray})
    data = Sigmoid.(x.value)
    inputs = [x]
    return Operation(inputs, data, z_grad::AbstractArray -> z_grad .* data .* (1 .- data), (x) -> Sigmoid.(x.value), ".Sigmoid")
end