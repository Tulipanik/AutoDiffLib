macro toposort(expr)
    return :(topological_sort($(esc(expr))))
end

function topological_sort(node::Node)
    visited = Set{Node}()
    order = Node[]
    _topological_sort!(node, visited, order)
    return order
end

function _topological_sort!(node::Node, visited::Set{Node}, order::Vector{Node})
    if node ∈ visited
        return
    end
    push!(visited, node)
    if node isa Operation || node isa SpreadedOperator
        for input in node.inputs
            _topological_sort!(input, visited, order)
        end
    end
    push!(order, node)
end