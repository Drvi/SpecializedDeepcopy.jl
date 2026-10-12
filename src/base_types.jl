# Preserve Base's special handling: share reflection objects, tasks and regexes; copy strings and
# big numbers; reset locks; reject modules.
const BaseCopied = Union{
    String, Symbol, Module, Task, Regex, GlobalRef, Method, Core.MethodInstance,
    BigInt, BigFloat, Base.AbstractLock,
}

specialized_deepcopy_internal(x::BaseCopied, copies) = Base.deepcopy_internal(x, copies)::typeof(x)

# From Julia 1.14, Base shares cancellation token sources and gives a wait entry, which has storage
# beyond its fields, a new entry from the runtime. A field copy of a wait entry would lack that storage.
@static if isdefined(Core, :CancellationTokenSource)
    specialized_deepcopy_internal(x::Core.CancellationTokenSource, copies) = Base.deepcopy_internal(x, copies)::typeof(x)
end
@static if isdefined(Core, :WaitEntryN)
    specialized_deepcopy_internal(x::Core.WaitEntryN, copies) = Base.deepcopy_internal(x, copies)::typeof(x)
end

# Reset the wait queue and copy the lock. get! preserves a condition copy registered by a cycle
# through the lock, keeping repeated conditions on one copied queue.
function specialized_deepcopy_internal(x::Base.GenericCondition, copies)
    haskey(copies, x) && return copies[x]::typeof(x)
    y = typeof(x)(_copy_value(getfield(x, :lock), copies))
    return get!(copies, x, y)::typeof(x)
end

specialized_deepcopy_internal(x::Type, copies) = x

# SimpleVector stores its elements outside its fields.
function specialized_deepcopy_internal(x::Core.SimpleVector, copies)
    haskey(copies, x) && return copies[x]::Core.SimpleVector
    y = Core.svec(Any[_copy_value(xi, copies)::typeof(xi) for xi in x]...)
    copies[x] = y
    return y
end

# Field reconstruction would bypass the garbage collector's WeakRef registration.
specialized_deepcopy_internal(::WeakRef, copies) = throw(PointerCopyError(WeakRef, nothing))

# The copy of a plain, String or Symbol key is equal to the key and hashes the same: a plain value
# bypasses custom methods, and a method for String or Symbol would be type piracy. So a Dict with
# such keys keeps its slots, and the copy replaces its keys and values in place. Base.deepcopy
# reinserts the entries unless the keys and values are bits types. Other keys can hash differently
# in the copy: a key can hash by identity or by its fields, a method can change a tuple that holds
# a type of its own, and a method can give a raw pointer another address. Those tables, and
# IdDicts, are rebuilt by reinserting the copied keys.
function _stable_hash(@nospecialize(K))
    (_is_plain(K) || K === String || K === Symbol) && return true
    return K isa Union && all(_stable_hash, Base.uniontypes(K))
end
@generated _stable(::Type{K}) where {K} = _stable_hash(K)

function specialized_deepcopy_internal(x::D, copies) where {D <: Union{Dict, IdDict}}
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
    if D <: Dict && !_settled(keytype(D))
        entries = sizehint!(Pair{keytype(D), valtype(D)}[], length(x))
        _copy_entries!(y, x, copies, entries)
        _check_later!(copies, y, entries)
    else
        _copy_entries!(y, x, copies, nothing)
    end
    return y
end

# A key that refers back to its table, directly or through other objects, enters the copied table
# before the copy has set all fields of the key and of the objects between them. Until then, its hash
# and its equality can read placeholders, so the key can land in a wrong slot, and two keys can compare
# equal for a moment, so that one entry replaces the other. So a Dict whose keys can change (see
# _settled_key) keeps its copied entries, and the top-level call refills the table from them after the
# copy has set every field, unless the table holds exactly those entries. A C function pointer for
# each table type keeps the list of tables free of runtime dispatch.
# The list is in `copies` under a mutable key: an IdDict hashes it by its address, which costs less
# in each top-level call than the hash of a singleton instance.
mutable struct _RebuiltTables end
const _REBUILT_TABLES = _RebuiltTables()

# A plain, String or Symbol key, or an immutable key whose fields have such types, is complete when
# the copied table receives it, and no mutable object is reachable from it. So its hash and its
# equality cannot change after the insertion, and its table needs no check. Any other type without
# fields, such as a raw pointer, can lead to objects outside its fields. A type that refers to itself
# through a Union field adds no other type.
function _settled_key(@nospecialize(K), seen = Any[])
    (_is_plain(K) || K === String || K === Symbol) && return true
    any(T -> T === K, seen) && return true
    push!(seen, K)
    K isa Union && return all(T -> _settled_key(T, seen), Base.uniontypes(K))
    return isconcretetype(K) && !ismutabletype(K) && fieldcount(K) > 0 && all(T -> _settled_key(T, seen), fieldtypes(K))
end
@generated _settled(::Type{K}) where {K} = _settled_key(K)

# An IdDict finds a key by its identity, which does not change during the copy.
_check_later!(copies, ::IdDict, entries) = nothing
function _check_later!(copies, table::D, entries) where {D <: Dict}
    tables = get!(Vector{Tuple{Ptr{Cvoid}, Any, Any}}, copies, _REBUILT_TABLES)::Vector{Tuple{Ptr{Cvoid}, Any, Any}}
    push!(tables, (_check_table_function_pointer(D), table, entries))
    return nothing
end

# Refill the table from its copied entries unless it holds exactly them: as many entries, each key
# found with its own value. Return whether it refilled the table.
function _check_table!(table::Dict, entries)
    length(table) == length(entries) && all(((k, v),) -> haskey(table, k) && table[k] === v, entries) && return false
    empty!(table)
    for (k, v) in entries
        table[k] = v
    end
    return true
end

# Defined after _check_table!: the generated @cfunction needs its binding, also when the package loads
# without compiled modules.
@generated function _check_table_function_pointer(::Type{D}) where {D}
    return :(@cfunction(_check_table!, Bool, (Ref{$D}, Ref{Vector{Pair{$(keytype(D)), $(valtype(D))}}})))
end

# A refilled table can change the keys of another table whose hash or equality reads it, so the passes
# repeat until none refills a table. The passes are limited to one more than the number of tables,
# which settles tables whose keys depend on other tables without a cycle. If the last pass still
# refills a table, the check raises an error instead of losing an entry: two copied keys of one table
# are equal, or the keys of tables depend on each other in a cycle that needs more passes.
function _check_tables!(copies)
    tables = get(copies, _REBUILT_TABLES, nothing)
    tables === nothing && return nothing
    list = tables::Vector{Tuple{Ptr{Cvoid}, Any, Any}}
    passes = length(list) + 1
    for _ in 1:passes
        refilled = false
        for (function_pointer, table, entries) in list
            refilled |= ccall(function_pointer, Bool, (Any, Any), table, entries)
        end
        if !refilled
            # A later call with the same `copies` checks only its own tables: a refill of a table from
            # this call would undo the changes that the caller made to it in between.
            delete!(copies, _REBUILT_TABLES)
            return nothing
        end
    end
    throw(ArgumentError("the check of copied Dicts refilled a Dict in all $passes passes that it allows: two " *
        "copied keys of one Dict are equal, or refilling a copied Dict changes the keys of another"))
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
# A vector `entries` also keeps each copied pair.
@generated function _copy_entries!(y, x, copies, entries)
    copy_key = _copy_expr(:k, keytype(x))
    copy_value = _copy_expr(:v, valtype(x))
    keep = entries === Nothing ? nothing : :(push!(entries, key => value))
    return quote
        for (k, v) in x
            key, value = $copy_key::typeof(k), $copy_value::typeof(v)
            y[key] = value
            $keep
        end
        return y
    end
end

# From Julia 1.11, an IdSet indexes its members by identity in fields of its own, and Base copies
# that index unchanged, so the copied set does not find its copied members. Reinserting them
# rebuilds the index. Before 1.11, an IdSet wraps an IdDict, which the generic field copy rebuilds
# while keeping aliases to it, as Base does.
@static if !hasfield(Base.IdSet{Any}, :dict)
    function specialized_deepcopy_internal(x::S, copies) where {S <: Base.IdSet}
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
    function specialized_deepcopy_internal(x::Array{T, N}, copies) where {T, N}
        haskey(copies, x) && return copies[x]::Array{T, N}
        y = Array{T, N}(undef, ntuple(_ -> 0, Val(N)))
        copies[x] = y
        setfield!(y, :ref, _copy_value(getfield(x, :ref), copies))
        setfield!(y, :size, getfield(x, :size))
        return y
    end

    function specialized_deepcopy_internal(x::MemoryRef{T}, copies) where {T}
        ref = memoryref(_copy_value(getfield(x, :mem), copies))
        i = Base.memoryrefoffset(x)
        # The initial reference also represents empty memory, where a bounds-checked move would fail.
        return i == 1 ? ref : Core.memoryrefnew(ref, i, true)
    end

    function specialized_deepcopy_internal(x::Memory{T}, copies) where {T}
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
    function specialized_deepcopy_internal(x::Array{T, N}, copies) where {T, N}
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
