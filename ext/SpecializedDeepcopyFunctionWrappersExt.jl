module SpecializedDeepcopyFunctionWrappersExt

using FunctionWrappers: FunctionWrapper
using SpecializedDeepcopy: SpecializedDeepcopy, specialized_deepcopy

# FunctionWrappers keeps the cconvert root in `obj` and the callable type in `objT`, both as Any.
# The root keeps the callable's storage alive; `objptr` addresses it and `ptr` points to compiled code.
# Register the wrapper before copying the root to preserve cycles. The Any fields require dispatch,
# so this is not trim-safe.
#
# Base's RefValue conversion uses the Ref's storage or the assigned callable's address. Custom
# roots or conversions may reach unfinished storage, so defer them until the first call. Setting
# `ptr` to C_NULL uses FunctionWrappers' reinit_wrapper path, which rebuilds both pointers without
# synchronization.
function SpecializedDeepcopy.specialized_deepcopy_internal(x::FunctionWrapper{R, A}, copies) where {R, A}
    haskey(copies, x) && return copies[x]::FunctionWrapper{R, A}
    y = ccall(:jl_new_struct_uninit, Any, (Any,), FunctionWrapper{R, A})::FunctionWrapper{R, A}
    copies[x] = y
    root = getfield(x, :obj)
    objT = getfield(x, :objT)
    copied = SpecializedDeepcopy._copy_value(root, copies)::typeof(root)
    setfield!(y, :obj, copied)
    setfield!(y, :objT, objT)
    # A cycle may reach an unfinished Ref. Mutable callables already have memoized copies; inline
    # callables use the Ref's stable storage. Defer boxed immutables: filling the Ref now would create
    # a box that the outer Ref copy later replaces.
    if base_conversion(objT, root) && (isdefined(copied, :x) || ismutabletype(objT) || Base.allocatedinline(objT))
        isdefined(copied, :x) || setfield!(copied, :x, SpecializedDeepcopy._copy_value(root[], copies))
        setfield!(y, :ptr, getfield(x, :ptr))
        setfield!(y, :objptr, convert(Ptr{Cvoid}, Base.unsafe_convert(Ref{objT}, copied)))
    else
        setfield!(y, :ptr, C_NULL)
        setfield!(y, :objptr, C_NULL)
    end
    return y
end

# Base's unsafe_convert(Ref{T}, root) delegates to unsafe_convert(Ptr{T}, root). Check both methods:
# a user override at either level may read storage that is still being copied.
function base_conversion(objT, root)
    root isa Base.RefValue{objT} || return false
    return which(Base.unsafe_convert, Tuple{Type{Ref{objT}}, typeof(root)}).module === Base &&
           which(Base.unsafe_convert, Tuple{Type{Ptr{objT}}, typeof(root)}).module === Base
end

end
