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

# CHOLMOD matrices and factorizations and UMFPACK factorizations own C memory through raw pointers.
# Where SparseArrays defines Base's method for them, it makes the copy with `copy`, which gives the
# copy C memory of its own. Older versions copy the addresses, and the generic copy here raises
# PointerCopyError instead.
_has_own_deepcopy(T, owner) = which(Base.deepcopy_internal, Tuple{T, IdDict}).module === owner
@static if _has_own_deepcopy(SparseArrays.CHOLMOD.Factor, SparseArrays.CHOLMOD)
    function SpecializedDeepcopy.specialized_deepcopy_internal(
            x::Union{SparseArrays.CHOLMOD.Dense, SparseArrays.CHOLMOD.Sparse, SparseArrays.CHOLMOD.Factor}, copies)
        return Base.deepcopy_internal(x, copies)::typeof(x)
    end
end
@static if _has_own_deepcopy(SparseArrays.UMFPACK.UmfpackLU, SparseArrays.UMFPACK)
    function SpecializedDeepcopy.specialized_deepcopy_internal(x::SparseArrays.UMFPACK.UmfpackLU, copies)
        return Base.deepcopy_internal(x, copies)::typeof(x)
    end
end

end
