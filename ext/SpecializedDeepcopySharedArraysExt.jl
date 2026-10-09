module SpecializedDeepcopySharedArraysExt

using SharedArrays: SharedArray
using SpecializedDeepcopy: SpecializedDeepcopy

# A SharedArray finds its data through an id and remote references to a shared-memory segment that
# its workers map. A field copy keeps them, so workers would write into the original's segment.
# Base's method makes a new segment on the same workers and copies the bits of the elements. Elements
# of a type that can hold a raw pointer then get the rule for Array elements: the method for their
# type, or PointerCopyError.
function SpecializedDeepcopy.specialized_deepcopy(x::SharedArray{T}, copies) where {T}
    haskey(copies, x) && return copies[x]::typeof(x)
    y = Base.deepcopy_internal(x, copies)::typeof(x)
    if !SpecializedDeepcopy._plain(T)
        for i in eachindex(y, x)
            y[i] = SpecializedDeepcopy._copy_value(x[i], copies)::T
        end
    end
    return y
end

end
