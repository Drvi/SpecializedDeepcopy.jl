module SpecializedDeepcopyPreallocationToolsExt

using PreallocationTools: DiffCache, FixedSizeDiffCache
using SpecializedDeepcopy: SpecializedDeepcopy

# PreallocationTools keeps get_tmp's workspaces in a field any_du::Vector{Any} or
# typed_du::Dict{DataType, Any}, depending on its version. Copying their Any entries needs runtime
# dispatch, so each original container gets one new empty container, memoized in `copies`: caches that
# share a container, such as a cache and its reshape, keep sharing one, and a container that the copy
# already reached through another reference keeps that copy. get_tmp allocates again on demand. Every
# other field, including a data buffer of type Vector{Any}, takes the generic path with its pointer rule
# and custom methods. As there, its copy must have the type of the original value, which can be
# narrower than the declared type of the field.
_is_workspace(name, F) = (name === :any_du && F === Vector{Any}) || (name === :typed_du && F === Dict{DataType, Any})

function _workspace_copy(x::W, copies) where {W}
    haskey(copies, x) && return copies[x]::W
    y = W()
    copies[x] = y
    return y
end

@generated function SpecializedDeepcopy.specialized_deepcopy(x::Union{DiffCache, FixedSizeDiffCache}, copies)
    fields = map(1:fieldcount(x)) do i
        F = fieldtype(x, i)
        v = :(getfield(x, $i))
        _is_workspace(fieldname(x, i), F) ? :(_workspace_copy($v, copies)) :
            :(let v = $v; SpecializedDeepcopy._copy_value(v, copies)::typeof(v) end)
    end
    return Expr(:new, x, fields...)
end

end
