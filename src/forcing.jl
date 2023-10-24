@inline get_arguments(forcing, particles, p, n) = [get_arg_value(particles[field], p, n) for field in forcing.field_dependencies if field in propertynames(particles)]

@inline get_arg_value(field::Number, args...) = field
@inline get_arg_value(field::Vector, p, n) = @inbounds get_arg_value(field[p], n)
@inline get_arg_value(field::Vector, n) = @inbounds field[n]
@inline get_arg_value(field::Matrix, n) = @inbounds field[n, :]