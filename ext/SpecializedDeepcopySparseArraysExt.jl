module SpecializedDeepcopySparseArraysExt

using SparseArrays: SparseArrays
using SpecializedDeepcopy: SpecializedDeepcopy

# From the SparseArrays of Julia 1.14, a CHOLMOD solve workspace owns C buffers through raw pointers.
# A copy of the addresses would not own the buffers, so the generic copy raises PointerCopyError.
# Base's method gives the copy a new, empty workspace, which allocates its own buffers in a solve.
@static if isdefined(SparseArrays.CHOLMOD, :CholmodWS)
    function SpecializedDeepcopy.specialized_deepcopy_internal(x::SparseArrays.CHOLMOD.CholmodWS, copies)
        return Base.deepcopy_internal(x, copies)::typeof(x)
    end
end

end
