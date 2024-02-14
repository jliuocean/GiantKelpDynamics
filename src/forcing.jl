@inline function get_arguments(forcing, particles, p, n)
    return_arguments = zeros(eltype(particles.position), length(forcing.field_dependencies))

    for (field_idx, field) in enumerate(forcing.field_dependencies)
        return_arguments[field_idx] = get_arg_value(particles[field], p, n)
    end

    return return_arguments
end

@inline get_arg_value(field::Number, args...) = field

@inline function get_arg_value(field, p, n) = 
    if length(size(field)) == 2
        return field[p, n]
    elseif length(size(field)) == 3
        return field[p, n, :]
    end
end