import Base: +, -, *, /, ^, sin, cos, tan, cot, sec, csc, exp, log, max

promote_to_node(x::Node) = x
promote_to_node(x::Number) = Constant(x)
promote_to_node(x::AbstractArray) = Constant(x)

# scalar methods
function +(x::Node{T}, y::Node{T}) where {T<:Number}
    data = x.value + y.value
    inputs = [x, y]
    return Operation(inputs, data, z_grad::T -> [z_grad, z_grad], "+")
end

+(x::Number, y::Node{T}) where {T<:Number} = promote_to_node(x) + y
+(x::Node{T}, y::Number) where {T<:Number} = x + promote_to_node(y)

function -(x::Node{T}, y::Node{T}) where {T<:Number}
    data = x.value - y.value
    inputs = [x, y]
    return Operation(inputs, data, z_grad::T -> [z_grad, -z_grad], "-")
end

-(x::Number, y::Node{T}) where {T<:Number} = promote_to_node(x) - y
-(x::Node{T}, y::Number) where {T<:Number} = x - promote_to_node(y)

function *(x::Node{T}, y::Node{T}) where {T<:Number}
    data = x.value * y.value
    inputs = [x, y]
    return Operation(inputs, data, z_grad::T -> [y.value * z_grad, x.value * z_grad], "*")
end

*(x::Number, y::Node{T}) where {T<:Number} = promote_to_node(x) * y
*(x::Node{T}, y::Number) where {T<:Number} = x * promote_to_node(y)

function /(x::Node{T}, y::Node{T}) where {T<:Number}
    data = x.value / y.value
    inputs = [x, y]
    return Operation(inputs, data, z_grad::T -> [z_grad / y.value, -(x.value * z_grad) / y.value^2], "/")
end

/(x::Number, y::Node{T}) where {T<:Number} = promote_to_node(x) / y
/(x::Node{T}, y::Number) where {T<:Number} = x / promote_to_node(y)

function ^(x::Node{T}, y::Node{T}) where {T<:Number}
    data = x.value ^ y.value
    inputs = [x, y]
    return Operation(inputs, data, z_grad::T -> [y.value*x.value^(y.value-1)*z_grad, log(x.value)*x.value^y.value*z_grad], "^")
end

^(x::Number, y::Node{T}) where {T<:Number} = promote_to_node(x) ^ y
^(x::Node{T}, y::Number) where {T<:Number} = x ^ promote_to_node(y)

function exp(x::Node{T}) where {T<:Number}
    data = exp(x.value)
    inputs = [x]
    return Operation(inputs, data, z_grad::T -> [z_grad*exp(x.value)], "exp")
end

function log(x::Node{T}) where {T<:Number}
    data = log(x.value)
    inputs = [x]
    return Operation(inputs, data, z_grad::T -> [z_grad*(1/x.value)], "log")
end

function sin(x::Node{T}) where {T<:Number}
    data = sin(x.value)
    inputs = [x]
    return Operation(inputs, data, z_grad::T -> [z_grad*cos(x.value)], "sin")
end

function cos(x::Node{T}) where {T<:Number}
    data = cos(x.value)
    inputs = [x]
    return Operation(inputs, data, z_grad::T -> [z_grad*-sin(x.value)], "cos")
end

function tan(x::Node{T}) where {T<:Number}
    data = tan(x.value)
    inputs = [x]
    return Operation(inputs, data, z_grad::T -> [z_grad/cos(x.value)^2], "tan")
end

function cot(x::Node{T}) where {T<:Number}
    data = cot(x.value)
    inputs = [x]
    return Operation(inputs, data, z_grad::T -> [-z_grad/sin(x.value)^2], "cot")
end

function sec(x::Node{T}) where {T<:Number}
    data = sec(x.value)
    inputs = [x]
    return Operation(inputs, data, z_grad::T -> [z_grad*sec(x.value)*tan(x.value)], "sec")
end

function csc(x::Node{T}) where {T<:Number}
    data = csc(x.value)
    inputs = [x]
    return Operation(inputs, data, z_grad::T -> [-z_grad*cot(x.value)*csc(x.value)], "csc")
end

function max(x::Node{T}, y::Node{T}) where {T<:Number}
    data = max(x.value, y.value)
    inputs = [x, y]
    if x.value > y.value
        return Operation(inputs, data, z_grad::T -> [z_grad, zero(T)], "max") # Use zero(T) for clarity
    elseif y.value > x.value
        return Operation(inputs, data, z_grad::T -> [zero(T), z_grad], "max") # Use zero(T) for clarity
    else
        return Operation(inputs, data, z_grad::T -> [z_grad/2, z_grad/2], "max")
    end
end

max(x::Number, y::Node{T}) where {T<:Number} = promote_to_node(x) |> (nx -> max(nx, y)) # Promote and then call the Node-Node method
max(x::Node{T}, y::Number) where {T<:Number} = promote_to_node(y) |> (ny -> max(x, ny)) # Promote and then call the Node-Node method

function Sigmoid(x::Node{T}) where {T<:Number}
    data = Sigmoid(x.value)
    inputs = [x]
    return Operation(inputs, data, z_grad::T -> [z_grad*data*(1-data)], "Sigmoid")
end

function ReLU(x::Node{T}) where {T<:Number}
    data = max(zero(x.value), x.value) # Assuming Sigmoid and ReLU operate on the value, not the Node itself
    inputs = [x]
    if x.value > 0
        return Operation(inputs, data, z_grad::T -> [z_grad], "ReLU")
    else
        return Operation(inputs, data, z_grad::T -> [zero(T)], "ReLU")
    end
end

function +(x::Node{T}, y::Node{T}) where {T<:AbstractArray}
    data = x.value .+ y.value
    inputs = [x, y]
    return Operation(inputs, data, z_grad::T -> [z_grad, z_grad], "+")
end

+(x::AbstractArray, y::Node{T}) where {T<:AbstractArray} = promote_to_node(x) + y
+(x::Node{T}, y::AbstractArray) where {T<:AbstractArray} = x + promote_to_node(y)

function -(x::Node{T}, y::Node{T}) where {T<:AbstractArray}
    data = x.value .- y.value
    inputs = [x, y]
    return Operation(inputs, data, z_grad::T -> [z_grad, -z_grad], "-")
end

-(x::AbstractArray, y::Node{T}) where {T<:AbstractArray} = promote_to_node(x) - y
-(x::Node{T}, y::AbstractArray) where {T<:AbstractArray} = x - promote_to_node(y)


function *(x::Node{<:AbstractArray}, y::Node{<:AbstractArray})
    data = x.value * y.value
    inputs = [x, y]
    return Operation(inputs, data, z_grad::AbstractArray -> [z_grad * y.value', x.value' * z_grad], "*")
end

*(x::AbstractArray, y::Node{T}) where {T<:AbstractArray} = promote_to_node(x) * y
*(x::Node{T}, y::AbstractArray) where {T<:AbstractArray} = x * promote_to_node(y)


function Base.Broadcast.broadcasted(::typeof(*), x::Node{<:AbstractArray}, y::Node{<:AbstractArray})
    data = x.value .* y.value
    inputs = [x, y]
    return Operation(inputs, data, z_grad::AbstractArray -> [z_grad .* y.value, z_grad .* x.value], ".*")
end

function Base.Broadcast.broadcasted(::typeof(*), x::Node{<:Number}, y::Node{<:AbstractArray})
    data = x.value .* y.value
    inputs = [x, y]
    return Operation(inputs, data, z_grad -> [sum(z_grad .* y.value), z_grad .* x.value], ".*")
end

function Base.Broadcast.broadcasted(::typeof(*), x::Node{<:AbstractArray}, y::Node{<:Number})
    data = x.value .* y.value
    inputs = Node[x, y]
    return Operation(inputs, data, z_grad -> [z_grad .* y.value, sum(z_grad .* x.value)], ".*")
end

function /(x::Node{<:AbstractArray}, y::Node{<:AbstractArray})
    data = x.value / y.value
    inputs = [x, y]
    Yinv = inv(y.value)
    return Operation(inputs, data, z_grad::AbstractArray -> [z_grad * Yinv', -x.value' * z_grad * Yinv'], "/")
end

/(x::AbstractArray, y::Node{T}) where {T<:AbstractArray} = promote_to_node(x) / y
/(x::Node{T}, y::AbstractArray) where {T<:AbstractArray} = x / promote_to_node(y)


function Base.Broadcast.broadcasted(::typeof(/), x::Node{<:AbstractArray}, y::Node{<:AbstractArray})
    data = x.value ./ y.value
    inputs = [x, y]
    return Operation(inputs, data, z_grad -> [z_grad ./ y.value, -z_grad .* x.value ./ (y.value .^ 2)], "./")
end

function Base.Broadcast.broadcasted(::typeof(/), x::Node{<:AbstractArray}, y::Node{<:Number})
    data = x.value ./ y.value
    inputs = [x, y]
    return Operation(inputs, data, z_grad -> [z_grad ./ y.value, -sum(z_grad .* x.value) / y.value^2], "./")
end

function Base.Broadcast.broadcasted(::typeof(/), x::Node{<:Number}, y::Node{<:AbstractArray})
    data = x.value ./ y.value
    inputs = [x, y]
    return Operation(inputs, data, z_grad -> [sum(z_grad ./ y.value), -z_grad .* x.value ./ (y.value .^ 2)], "./")
end