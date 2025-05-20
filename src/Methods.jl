import Base: +, -, *, /

function +(x::Node{T}, y::Node{T}) where {T <: Number}
    data = x.value + y.value
    inputs = [x, y]
    function backward(z_grad::T)
        return [z_grad, z_grad]
    end
    return Operation(inputs, data, backward, "+")
end

function -(x::Node{T}, y::Node{T}) where {T <: Number}
    data = x.value - y.value
    inputs = [x, y]
    function backward(z_grad::T)
        return [z_grad, -z_grad]
    end
    return Operation(inputs, data, backward, "-")
end

function Base.:+(x::Node{T}, y::Node{T}) where {T <: AbstractArray}
    data = x.value .+ y.value
    inputs = [x, y]
    function backward(z_grad::T)
        return [z_grad, z_grad]
    end
    return Operation(inputs, data, backward, "+")
end

# function Base.:+(x::Node, y::Node)
#     data = x.value .+ y.value
#     inputs = [x, y]
#     return Operation("+", inputs, data, zero(data), σ-> [σ, σ])
# end

# function Base.:-(x::Node, y::Node)
#     data = x.value .- y.value
#     inputs = [x, y]
#     function backward(z_grad::AbstractArray{Float64})
#         return [z_grad, -z_grad]
#     end
#     return Operation("-", inputs, data, zero(data), backward)
# end

# function Base.sum(x::Node; dims=:)
#     data = sum(x.value, dims=dims)
#     inputs = [x]
#     function backward(z_grad::AbstractArray{Float64})
#         return [fill(z_grad, size(x.value))]
#     end
#     return Operation("sum", inputs, data, zero(data), backward)
# end