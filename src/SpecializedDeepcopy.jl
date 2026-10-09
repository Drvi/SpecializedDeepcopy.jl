module SpecializedDeepcopy

export specialized_deepcopy, PointerCopyError

"""
    specialized_deepcopy(x)

Copy `x` using generated code for its concrete type. The target contract is `Base.deepcopy` with
stricter pointer handling; the current exceptions are described below and in the README.

Concrete fields and elements, or supported small unions, can avoid runtime dispatch. Custom methods
and inference limits can still prevent compilation with `juliac --trim=safe`.

Copy related values as a tuple to preserve sharing between them:

```julia
weights = [1.0, 2.0]
a, b = specialized_deepcopy((weights, weights))
@assert a === b && a !== weights
```

Base copies raw pointer addresses unchanged. Here a raw pointer (`Ptr`, `Cstring`, `Cwstring` or
`Core.LLVMPtr`) or `WeakRef` raises [`PointerCopyError`](@ref), unless an enclosing type's method
handles ownership. The FunctionWrappers extension rebuilds the copied callable's object pointer.

Non-pointer exceptions include rebuilding `IdSet` tables to avoid Base's member-lookup bug on
Julia 1.11–1.13, resetting PreallocationTools workspaces, and the custom-method rules below.

    specialized_deepcopy(x, copies::IdDict{Any,Any})

Add a two-argument method for your type. Custom `Base.deepcopy_internal` methods do not apply to
the generic field copier. `copies` maps originals to copies, some still under construction:

1. Return a value of the same type as `x`.
2. Copy each nested value with `specialized_deepcopy(value, copies)`.
3. If the type is mutable, return `copies[x]` when `haskey(copies, x)`. Set `copies[x]` to the copy
   before you copy a field that can lead back to `x`.

Do not inspect unfinished copies retrieved from `copies`. A trim-safe method must avoid runtime
dispatch. A one-argument method changes only top-level calls.

A plain value is an `isbitstype` value containing no raw pointers; generic copying reuses it without
calling a custom method. Constructor-optional fields also bypass custom methods when their declared
type is concrete, inline, immutable, and contains neither references nor raw pointers. This applies
even to initialized fields: unset storage can hold invalid Union tags without a definedness flag.

For example, explicitly share a lookup table:

```julia
struct LookupTable
    values::Vector{Float64}
end
SpecializedDeepcopy.specialized_deepcopy(x::LookupTable, copies) = x
```
"""
function specialized_deepcopy end

specialized_deepcopy(x) = _copy_value(x, IdDict{Any, Any}())::typeof(x)
specialized_deepcopy(x, copies) = _copy_by_fields(x, copies)

# Generic copy paths bypass customization for pointer-free bits values.
_copy_value(x::T, copies) where {T} = _plain(T) ? x : specialized_deepcopy(x, copies)

# Bound inference by nested type parameters, not by object traversal depth.
const MAX_INFERRED_TYPE_DEPTH = 16

# Julia's internal recursion_relation hook bypasses a soft recursion limit for `_copy_value`
# while its value type's parameter depth is within the bound. Other inference limits still apply.
# Without the hook, inference can widen nested copy calls and leave them unresolved under --trim.
# _copy_expr splits unions separately because inference's own union splitting can hit a hard limit
# without consulting this hook.
# Depth counts type parameters: Float64 has depth 1, Tuple{Vector{Float64}} has depth 3, and a
# non-parametric linked-list type stays at depth 1. test/trim/shapes.jl covers the boundary;
# test/runtests.jl checks that precompilation preserves the hook.
function _copy_recursion_ends(method, callee_method, @nospecialize(sig), @nospecialize(earlier_sig))
    s = Base.unwrap_unionall(sig)
    s isa DataType && length(s.parameters) >= 2 || return false
    return _type_depth(s.parameters[2]) <= MAX_INFERRED_TYPE_DEPTH
end

function _type_depth(@nospecialize(t))
    t isa Union && return max(_type_depth(t.a), _type_depth(t.b))
    t isa UnionAll && return _type_depth(t.body)
    t isa DataType || return 0
    return 1 + maximum(_type_depth, t.parameters; init = 0)
end

# Attach to the method so precompilation preserves the hook.
for m in methods(_copy_value)
    m.recursion_relation = _copy_recursion_ends
end

"""
    PointerCopyError(type, field)

Copying a value of `type` encountered a raw pointer (`Ptr`, `Cstring`, `Cwstring` or `Core.LLVMPtr`)
or `WeakRef`. `field` names the raw pointer field, or is `nothing` for a bare pointer or `WeakRef`.

A copied pointer does not retain the storage's owner. A field-by-field `WeakRef` copy is not
registered with the garbage collector. Either can dangle after the original storage is freed.

Add `SpecializedDeepcopy.specialized_deepcopy(x::T, copies)` for the enclosing type to define safe
ownership, for example by sharing `x` or duplicating its storage.
"""
struct PointerCopyError <: Exception
    type::Type
    field::Union{Symbol, Int, Nothing}
end

function Base.showerror(io::IO, e::PointerCopyError)
    print(io, "PointerCopyError: specialized_deepcopy cannot copy a value of type ", e.type, ". ")
    if e.field === nothing
        print(io, "The value is a raw pointer or a WeakRef. ")
    else
        print(io, "Its field `", e.field, "` is a raw pointer of type ", fieldtype(e.type, e.field), ". ")
    end
    print(io, "A copy of it refers to memory that the copy does not keep alive. ")
    if e.field === nothing
        print(io, "Add a method `SpecializedDeepcopy.specialized_deepcopy(x::T, copies)` for the type T that holds the value.")
    else
        print(io, "Add a method `SpecializedDeepcopy.specialized_deepcopy(x::T, copies)` that shares or duplicates the value, ",
            "with T = ", e.type, " if your code defines that type, or else a type of your own that holds the value.")
    end
end

const RawPointer = Union{Ptr, Core.LLVMPtr, Cstring, Cwstring}

# `Union{}` is a subtype of every type, but a field of that type never holds a value.
_is_raw_pointer(@nospecialize(T)) = T !== Union{} && T <: RawPointer

# The bits of a plain value are a complete copy of it.
_is_plain(@nospecialize(T)) = isbitstype(T) && !_holds_raw_pointer(T)
function _holds_raw_pointer(@nospecialize(T))
    T === Union{} && return false
    T isa Union && return any(_holds_raw_pointer, Base.uniontypes(T))
    return _is_raw_pointer(T) || any(_holds_raw_pointer, fieldtypes(T))
end

# Optional inline fields without references have no definedness flag. Their unset bits may contain
# invalid Union tags, so transfer them without traversing fields or calling a custom method.
function _copies_bits(T::DataType, i::Int)
    i > fieldcount(T) - T.name.n_uninitialized || return false
    F = fieldtype(T, i)
    return isconcretetype(F) && !ismutabletype(F) && Base.allocatedinline(F) && Base.datatype_pointerfree(F) &&
        !_holds_raw_pointer(F)
end

# Emit a literal Bool for each type so inference can resolve the branch.
@generated _plain(::Type{T}) where {T} = _is_plain(T)

# Only constructor-omitted trailing fields need a definedness check. Non-atomic bits types and
# isbits unions are always readable. Keep the check for all atomic fields: some layouts are boxed.
# Inline structs containing references can also be undefined.
function _may_be_undefined(T::DataType, i::Int)
    F = fieldtype(T, i)
    inline = (isbitstype(F) || Base.isbitsunion(F)) && !Base.isfieldatomic(T, i)
    return i > fieldcount(T) - T.name.n_uninitialized && !inline
end

_construct_expr(T::DataType, args) = T <: Tuple ? Expr(:tuple, args...) : Expr(:new, T, args...)

# Fieldless runtime types with variable-sized storage, such as SimpleVector, need dedicated copies.
function _has_hidden_contents(T::DataType)
    (fieldcount(T) == 0 && !isbitstype(T)) || return false
    try
        sizeof(T)
        return false
    catch
        return true
    end
end

# Bound the generated branches and calls for declared unions and tuples. Larger sets use an unsplit
# call, which can require runtime dispatch.
const MAX_SPLIT_TYPES = 16

# Enumerate up to MAX_SPLIT_TYPES concrete value types, or return nothing.
# Tuple covariance matters: a Tuple{Union{Missing, Int}} value has type Tuple{Missing} or Tuple{Int}.
function _concrete_types(@nospecialize(F))
    isconcretetype(F) && return Any[F]
    if F isa Union
        parts = map(_concrete_types, Base.uniontypes(F))
        any(isnothing, parts) && return nothing
        types = reduce(vcat, parts)
    elseif F isa DataType && F <: Tuple && !any(Base.isvarargtype, F.parameters)
        parts = map(_concrete_types, F.parameters)
        any(isnothing, parts) && return nothing
        count = 1
        for part in parts
            # The product is checked at each step, so it cannot overflow.
            count *= length(part)
            count <= MAX_SPLIT_TYPES || return nothing
        end
        types = vec(Any[Tuple{combination...} for combination in Iterators.product(parts...)])
    else
        return nothing
    end
    return length(types) <= MAX_SPLIT_TYPES ? types : nothing
end

# Split finite declared unions and tuples into concrete calls. This handles tuple covariance and
# avoids inference's union-splitting recursion limit, notably before Julia 1.13. A field such as
# next::Union{Nothing, Node} must retain its recursive call under --trim.
function _copy_expr(v::Symbol, @nospecialize(F))
    copy = :(_copy_value($v, copies))
    isconcretetype(F) && return copy
    types = _concrete_types(F)
    types === nothing && return copy
    split = :(throw(TypeError(:specialized_deepcopy, "declared type", $F, $v)))
    for U in reverse(types)
        split = :($v isa $U ? $copy : $split)
    end
    return split
end

# Write fields directly, as Base.deepcopy does; constructors and their checks do not run.
function _copy_by_fields_expr(T::DataType)
    _is_raw_pointer(T) && return :(throw(PointerCopyError($T, nothing)))
    _is_plain(T) && return :x
    for i in 1:fieldcount(T)
        if _is_raw_pointer(fieldtype(T, i))
            return :(throw(PointerCopyError($T, $(QuoteNode(fieldname(T, i))))))
        end
    end
    if _has_hidden_contents(T)
        message = "specialized_deepcopy has no method for $T, whose contents are not fields"
        return :(throw(ArgumentError($message)))
    end
    ismutabletype(T) && return _mutable_copy_expr(T)
    return _immutable_copy_expr(T)
end

# An atomic field can only be read and written with an ordering.
_order(T::DataType, i::Int) = Base.isfieldatomic(T, i) ? (QuoteNode(:sequentially_consistent),) : ()

function _mutable_copy_expr(T::DataType)
    # n_uninitialized counts trailing fields that some constructor leaves unset.
    initialized = fieldcount(T) - T.name.n_uninitialized
    # Seed required fields with original values: the interpreter rejects `new` with fewer arguments.
    # Register the shell before recursing, then replace non-plain fields with their copies. A cycle
    # may observe original values in required fields not yet replaced, or unset optional fields.
    placeholders = [:(getfield(x, $i, $(_order(T, i)...))) for i in 1:initialized]
    body = Expr(:block,
        :(haskey(copies, x) && return copies[x]::$T),
        :(y = $(Expr(:new, T, placeholders...))),
        :(copies[x] = y),
    )
    for i in 1:fieldcount(T)
        F = fieldtype(T, i)
        # A field of type `Union{}` never holds a value.
        (F === Union{} || (i <= initialized && _is_plain(F))) && continue
        order = _order(T, i)
        v, c = Symbol(:v, i), Symbol(:c, i)
        # setfield! rejects const fields; use Base.deepcopy's runtime setter to initialize them.
        set = if isconst(T, i)
            :(ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), y, $(i - 1), $c))
        else
            :(setfield!(y, $i, $c, $(order...)))
        end
        step = quote
            $v = getfield(x, $i, $(order...))
            $c = $(_copies_bits(T, i) ? v : _copy_expr(v, F))::typeof($v)
            $set
        end
        if _may_be_undefined(T, i)
            step = :(if isdefined(x, $i, $(order...)); $step; end)
        end
        push!(body.args, step)
    end
    push!(body.args, :(return y))
    return body
end

function _immutable_copy_expr(T::DataType)
    body = Expr(:block)
    copied = Symbol[]
    for i in 1:fieldcount(T)
        # An uninhabited field ends the defined prefix. Avoid isdefined here: after another field's
        # definedness check, it can make Julia 1.13 inference fail.
        if fieldtype(T, i) === Union{}
            push!(body.args, :(return $(_construct_expr(T, copied))))
            return body
        end
        if _may_be_undefined(T, i)
            # Like Base.deepcopy, reconstruct only the defined prefix.
            push!(body.args, :(isdefined(x, $i) || return $(_construct_expr(T, copied))))
        end
        v, c = Symbol(:v, i), Symbol(:c, i)
        copy = _copies_bits(T, i) ? v : _copy_expr(v, fieldtype(T, i))
        push!(body.args, :($v = getfield(x, $i)), :($c = $copy::typeof($v)))
        push!(copied, c)
    end
    push!(body.args, :(return $(_construct_expr(T, copied))))
    return body
end

@generated function _copy_by_fields(x::T, copies) where {T}
    return _copy_by_fields_expr(T)
end

include("base_types.jl")

end # module
