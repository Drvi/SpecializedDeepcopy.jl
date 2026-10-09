# Preserve Base's special handling: share reflection objects, tasks and regexes; copy strings and
# big numbers; reset locks; reject modules.
const BaseCopied = Union{
    String, Symbol, Module, Task, Regex, GlobalRef, Method, Core.MethodInstance,
    BigInt, BigFloat, Base.AbstractLock,
}

specialized_deepcopy(x::BaseCopied, copies) = Base.deepcopy_internal(x, copies)::typeof(x)

# Reset the wait queue and copy the lock. get! preserves a condition copy registered by a cycle
# through the lock, keeping repeated conditions on one copied queue.
function specialized_deepcopy(x::Base.GenericCondition, copies)
    haskey(copies, x) && return copies[x]::typeof(x)
    y = typeof(x)(_copy_value(getfield(x, :lock), copies))
    return get!(copies, x, y)::typeof(x)
end

specialized_deepcopy(x::Type, copies) = x

# SimpleVector stores its elements outside its fields.
function specialized_deepcopy(x::Core.SimpleVector, copies)
    haskey(copies, x) && return copies[x]::Core.SimpleVector
    y = Core.svec(Any[_copy_value(xi, copies)::typeof(xi) for xi in x]...)
    copies[x] = y
    return y
end

# Field reconstruction would bypass the garbage collector's WeakRef registration.
specialized_deepcopy(::WeakRef, copies) = throw(PointerCopyError(WeakRef, nothing))

# A key of one of these types hashes by value, so its copy lands in the same slot and the table's
# slots are copied with the entries, as Base.deepcopy does. A mutable key hashes by identity, and
# so does an IdDict key, so those tables are rebuilt by reinserting copied keys. So is a table whose
# keys can hold a raw pointer: the method of such a key can give its copy another pointer.
function _stable_hash(@nospecialize(K))
    (_is_plain(K) || K === String || K === Symbol) && return true
    K isa Union && return all(_stable_hash, Base.uniontypes(K))
    return isconcretetype(K) && (K <: Tuple || K <: NamedTuple) && all(_stable_hash, fieldtypes(K))
end
@generated _stable(::Type{K}) where {K} = _stable_hash(K)

function specialized_deepcopy(x::D, copies) where {D <: Union{Dict, IdDict}}
    haskey(copies, x) && return copies[x]::D
    if _plain(keytype(D)) && _plain(valtype(D))
        y = copy(x)
        copies[x] = y
        return y
    end
    if D <: Dict && _stable(keytype(D))
        y = copy(x)
        copies[x] = y
        return _copy_slots!(y, copies)
    end
    y = empty(x)
    copies[x] = y
    return _copy_entries!(y, x, copies)
end

# Replace the keys and values of the filled slots in place; Base.isslotfilled reads the slot byte.
@generated function _copy_slots!(y, copies)
    K, V = keytype(y), valtype(y)
    body = Expr[]
    _is_plain(K) || K === Symbol || push!(body, :(k = keys[i]; keys[i] = $(_copy_expr(:k, K))::typeof(k)))
    _is_plain(V) || push!(body, :(v = vals[i]; vals[i] = $(_copy_expr(:v, V))::typeof(v)))
    return quote
        keys, vals = y.keys, y.vals
        for i in eachindex(vals)
            Base.isslotfilled(y, i) || continue
            $(body...)
        end
        return y
    end
end

# Generate these loops so _copy_expr can split declared key, value and element types, as for fields.
@generated function _copy_entries!(y, x, copies)
    copy_key = _copy_expr(:k, keytype(x))
    copy_value = _copy_expr(:v, valtype(x))
    return quote
        for (k, v) in x
            y[$copy_key::typeof(k)] = $copy_value::typeof(v)
        end
        return y
    end
end

# On Julia 1.11–1.13, Base copies IdSet's members but retains their old identity-based index.
# Reinserting the copied members rebuilds that index so lookups work.
function specialized_deepcopy(x::S, copies) where {S <: Base.IdSet}
    haskey(copies, x) && return copies[x]::S
    y = S()
    copies[x] = y
    return _copy_members!(y, x, copies)
end

@generated function _copy_members!(y, x, copies)
    copy = _copy_expr(:k, eltype(x))
    return quote
        for k in x
            push!(y, $copy::typeof(k))
        end
        return y
    end
end

@generated function _copy_elements!(y, x, copies)
    copy = _copy_expr(:xi, eltype(x))
    return quote
        for i in eachindex(y, x)
            if isassigned(x, i)
                xi = x[i]
                y[i] = $copy::typeof(xi)
            end
        end
        return y
    end
end

@static if isdefined(Base, :Memory)
    # Since Julia 1.11, Array stores its shape and a MemoryRef in `size` and `ref`. Copying the ref
    # preserves shared backing Memory and offsets. Register the empty array first so elements can
    # refer back to it, as in Base.deepcopy.
    function specialized_deepcopy(x::Array{T, N}, copies) where {T, N}
        haskey(copies, x) && return copies[x]::Array{T, N}
        y = Array{T, N}(undef, ntuple(_ -> 0, Val(N)))
        copies[x] = y
        setfield!(y, :ref, _copy_value(getfield(x, :ref), copies))
        setfield!(y, :size, getfield(x, :size))
        return y
    end

    function specialized_deepcopy(x::MemoryRef{T}, copies) where {T}
        ref = memoryref(_copy_value(getfield(x, :mem), copies))
        i = Base.memoryrefoffset(x)
        # The initial reference also represents empty memory, where a bounds-checked move would fail.
        return i == 1 ? ref : Core.memoryrefnew(ref, i, true)
    end

    function specialized_deepcopy(x::Memory{T}, copies) where {T}
        haskey(copies, x) && return copies[x]::Memory{T}
        y = _plain(T) ? copy(x) : Memory{T}(undef, length(x))
        copies[x] = y
        if !_plain(T)
            # Every mutable element gets an entry; reserving them avoids repeated rehashing.
            isconcretetype(T) && ismutabletype(T) && sizehint!(copies, length(copies) + length(x))
            _copy_elements!(y, x, copies)
        end
        return y
    end
else
    # Julia 1.10: copy each array's buffer separately, matching Base.deepcopy for reshape aliases.
    function specialized_deepcopy(x::Array{T, N}, copies) where {T, N}
        haskey(copies, x) && return copies[x]::Array{T, N}
        y = _plain(T) ? copy(x) : similar(x)
        copies[x] = y
        if !_plain(T)
            # Every mutable element gets an entry; reserving them avoids repeated rehashing.
            isconcretetype(T) && ismutabletype(T) && sizehint!(copies, length(copies) + length(x))
            _copy_elements!(y, x, copies)
        end
        return y
    end
end
