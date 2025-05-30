abstract type Node{T} end

mutable struct Variable{T} <: Node{T}
    value::T
    grad::T
    name::String
    Variable(val::T, name::String="") where {T <: AbstractArray} = new{AbstractArray{Float32}}(Float32.(val), Float32.(zero(val)), name)
    Variable(val::T, name::String="") where {T <: Number} = new{Float32}(Float32(val), Float32(zero(val)), name)
end

struct Constant{T} <: Node{T}
    value::T
    Constant(val::T) where {T <: AbstractArray} = new{AbstractArray{Float32}}(Float32.(val))
    Constant(val::T) where {T <: Number} = new{Float32}(Float32(val))
end

mutable struct Operation{T} <: Node{T}
    op_name::String
    inputs::Vector{<:Node}
    value::T
    grad::T
    backward::Function
    foward::Function
    Operation(inputs::Vector{<:Node}, value::T, backward::Function, foward::Function, op_name::String="") where {T <: Number} = new{Float32}(op_name, inputs, Float32(value), Float32(zero(value)), backward, foward)
    Operation(inputs::Vector{<:Node}, value::T, backward::Function, foward::Function, op_name::String="") where {T <: AbstractArray} = new{AbstractArray{Float32}}(op_name, inputs, Float32.(value), Float32.(zeros(size(value))), backward, foward)
end

mutable struct SpreadedOperator{T} <: Node{T}
    op_name::String
    inputs::Vector{<:Node}
    value::T
    grad::AbstractArray{T}
    backward::Function
    foward::Function
    SpreadedOperator(inputs::Vector{<:Node}, value::T, backward::Function, foward::Function, op_name::String="") where {T <: Number} = new{Float32}(op_name, inputs, Float32(value), zeros(Float32, size(inputs[1].value)), backward, foward)
end

