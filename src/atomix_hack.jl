using Oceananigans: Field

import Atomix: pointer

@inline function pointer(ref::Atomix.Internal.IndexableRef{<:Field, Tuple{Vararg{Int64, N}}} where {N, Indexable})
    i = LinearIndices(ref.data.data)[ref.indices...]
    return Base.pointer(ref.data.data, i)
end