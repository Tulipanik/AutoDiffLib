abstract type Node{T} end

mutable struct Variable{T} <: Node{T}
    value::T
    grad::T
    name::String
    Variable(val::T, name::String="") where {T <: AbstractArray} = new{T}(val, zero(val), name)
    Variable(val::T, name::String="") where {T <: Number} = new{T}(val, zero(val), name)
end

struct Constant{T} <: Node{T}
    value::T
end

mutable struct Operation{T} <: Node{T}
    op_name::String
    inputs::Vector{Node}
    value::T
    grad::T
    backward::Function
    Operation(inputs::Vector{<:Node{T}}, value::T, backward::Function, op_name::String="") where {T <: Number} = new{T}(op_name, inputs, value, zero(value), backward)
    Operation(inputs::Vector{<:Node{T}}, value::T, backward::Function, op_name::String="") where {T <: AbstractArray} = new{T}(op_name, inputs, value, zeros(size(value)), backward)
    # Operation(inputs::Vector{<:Node{T1}}, value::T2, backward::Function, op_name::String="") where {T1 <: Number, T2 <: AbstractArray} = new{T}(op_name, inputs, value, zeros(size(value)), backward)
    # Operation(inputs::Vector{<:Node}, value::T, backward::Function, op_name::String="") where {T <: AbstractArray} = new{T}(op_name, inputs, value, zeros(size(value)), backward)
end
