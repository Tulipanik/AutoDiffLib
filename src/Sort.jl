function topological_sort(node::Node{T}, visited::Set{Node{T}}, order::Vector{Node{T}}) where {T}
    if node in visited
        return
    end
    push!(visited, node)
    if node isa Operation{T}
        for input in node.inputs
            topological_sort(input, visited, order)
        end
    end
    push!(order, node)
    return order
end