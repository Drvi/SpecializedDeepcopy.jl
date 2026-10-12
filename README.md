# SpecializedDeepcopy

`specialized_deepcopy(x)` generates copy code for concrete types so suitable object graphs can be copied in programs built with `juliac --trim=safe`. `Base.deepcopy` uses runtime dispatch that the trim verifier rejects. Because `specialized_deepcopy` specializes on every type, expect longer compile times and larger code than with `Base.deepcopy`.

The target contract is **`Base.deepcopy` with stricter pointer handling**: preserve types, values, aliases and cycles, with an explicit ownership policy for pointer-backed state. The current implementation has the non-pointer differences listed below.

Supports Julia 1.10–1.13. Trim tests require Julia 1.12 or later.

## Usage

```julia
using SpecializedDeepcopy

x = (weights = [1.0, 2.0],)
y = specialized_deepcopy(x)
@assert typeof(y) === typeof(x) && y.weights == x.weights
@assert y.weights !== x.weights
```

Copy related values together to preserve their shared objects:

```julia
weights = [1.0, 2.0]
inputs, outputs = specialized_deepcopy((weights, weights))
@assert inputs === outputs && inputs !== weights
```

To preserve sharing between values that you cannot put in one tuple, pass the same `IdDict{Any,Any}` to several calls. It maps each original to its copy:

```julia
weights = [1.0, 2.0]
copies = IdDict{Any,Any}()
inputs = specialized_deepcopy(weights, copies)
outputs = specialized_deepcopy((weights,), copies)
@assert outputs[1] === inputs
```

Each call completes the copied dictionaries of that call whose keys can change during the copy; see [Different from Base.deepcopy](#different-from-basedeepcopy). If a call throws, `copies` holds unfinished copies, so do not pass it to another call. [Custom copy methods](#custom-copy-methods) extend `specialized_deepcopy_internal`, which copies the values inside a call.

Do not mutate the input graph during copying.

## Differences from Base.deepcopy

| Value | `Base.deepcopy` | `specialized_deepcopy` |
| --- | --- | --- |
| Raw pointer (`Ptr`, `Cstring`, `Cwstring`, `Core.LLVMPtr`) | Copies the address unchanged | Throws `PointerCopyError` unless an enclosing type's method handles ownership |
| `WeakRef` | Copies its fields | Throws `PointerCopyError`; field copying bypasses GC registration |
| `WeakKeyDict` | Rebuilds the table with copied keys | Throws `PointerCopyError`, because the table holds its keys as `WeakRef`s |
| `FunctionWrapper` | Retains the original object pointer | The extension copies the callable storage and rebuilds the pointer |
| `DiffCache` / `FixedSizeDiffCache` workspaces | Copies existing workspaces | The extension starts with empty workspace containers |
| `IdSet` | From Julia 1.11, the copied set does not find most of its members | From Julia 1.11, rebuilds the set, so lookups work |
| Optional non-isbits inline field described below | Calls the value's copy method | Copies its bits without a method call |

The `IdSet` and `FunctionWrapper` rows fix defects of `Base.deepcopy`. From Julia 1.11, `Base.deepcopy` keeps the stale objectids of the original members in the index of the copied `IdSet`. With a `FunctionWrapper`, `Base.deepcopy` does not deep-copy a stateful callable properly: calls on the copy still use the state of the original.

Copying pointer bits neither copies nor keeps the target alive. A custom method must decide whether to share or duplicate the resource. A union such as `Union{Nothing, Ptr{Cvoid}}` is rejected only when it holds a pointer. A `WeakKeyDict` needs a method on the type that holds it, for example one that starts an empty table, as [test/examples/result_cache.jl](test/examples/result_cache.jl) shows.

The last row applies to fields that a constructor can leave unset, with a concrete, immutable, inline type containing no references or raw pointers. It applies **even when the field is initialized**. Unset bits can contain invalid Union tags, and there is no definedness flag to distinguish them. For a non-plain type `T`, declaring the field as `Union{T, Nothing}` allows its custom method to run.

`specialized_deepcopy` does not call custom `Base.deepcopy_internal` methods of other packages. A type with such a method needs a `specialized_deepcopy_internal` method too. [Custom copy methods](#custom-copy-methods) describes how to write one.

## Behavior shared with Base.deepcopy

Structs are copied field by field without calling constructors. Declared types, const and atomic fields, and legitimately undefined fields are preserved. Mutable copies are registered before their fields are copied, allowing cycles.

A **plain value** is an `isbitstype` value containing no raw pointer. It is returned unchanged without calling a custom method.

Special handling covers:

- `Array`, `Memory`, `MemoryRef`: new storage. On Julia 1.11+, shared array storage stays shared within the copy. Julia 1.10 copies each array separately, as Base does.
- `Dict` with plain, `String` or `Symbol` keys, or a union of these: the copy keeps the slots and replaces the keys and values, because the copy of such a key hashes the same. Other `Dict`s and `IdDict`: rebuilt tables with the copied keys. For a key whose hash changes during the copy, see [Different from Base.deepcopy](#different-from-basedeepcopy).
- `Core.SimpleVector`: copied elements.
- Strings, big numbers and locks: the methods of `Base.deepcopy`. Conditions get a copied lock and an empty wait queue.
- Symbols, types, tasks, regular expressions, methods, method instances and `GlobalRef`: shared with the original.
- Modules: rejected, as by Base.

Other identity-based collections may need a method that reinserts copied keys.

## Custom copy methods

Define `SpecializedDeepcopy.specialized_deepcopy_internal(x::YourType, copies)`. The `copies::IdDict{Any,Any}` argument maps originals to copies, some still under construction.

In version 0.1, custom methods extended `specialized_deepcopy` with two arguments. To update such code, rename these methods, and their calls inside custom methods, to `specialized_deepcopy_internal`. Keep the other calls on `specialized_deepcopy`. A method that keeps the old name no longer runs for nested values, and copying a value of its type directly raises a `MethodError`.

Custom `Base.deepcopy_internal` methods of other packages must be ported to this API. Base and the standard libraries need no port: for each of their types with such a method, the package has a method of its own or calls Base's method. For pointers in Base containers, such as `Vector{Ptr{Cvoid}}`, define the method on your enclosing type.

1. Return the same type as `x`.
2. Use `specialized_deepcopy_internal(value, copies)` for nested values. Never call `specialized_deepcopy` from `specialized_deepcopy_internal`, you may only call the internal version recursively.
3. For mutable types, return an existing `copies[x]`, or register the new copy before following a reference that could lead back to `x`.
4. Do not read fields of an unfinished copy obtained from `copies`.

For example, deliberately share a lookup table:

```julia
struct LookupTable
    values::Vector{Float64}
end
SpecializedDeepcopy.specialized_deepcopy_internal(x::LookupTable, copies) = x
```

Rebuild an index whose keys depend on object identity:

```julia
mutable struct Owner
    id::Int
end
struct OwnerIndex
    owners::Vector{Owner}
    slots::Dict{UInt, Int}
end
function OwnerIndex(owners)
    slots = Dict(objectid(owner) => i for (i, owner) in pairs(owners))
    return OwnerIndex(owners, slots)
end

function SpecializedDeepcopy.specialized_deepcopy_internal(x::OwnerIndex, copies)
    return OwnerIndex(specialized_deepcopy_internal(x.owners, copies))
end
```

For externally allocated memory, see `ExternBuffer` in [test/examples/externally_allocated_memory.jl](test/examples/externally_allocated_memory.jl). Its method allocates independent storage, preserves both owners while copying bytes, and installs a finalizer.

### FunctionWrappers

Loading FunctionWrappers.jl enables copying of callable storage and reconstruction of the wrapper's object pointer. [test/examples/wrapped_mutable_callable.jl](test/examples/wrapped_mutable_callable.jl) shows a callable with state: calls on the copy that `Base.deepcopy` makes change the original, and calls on the copy that this package makes do not. The extension requires FunctionWrappers 1.1.2 or later. Earlier versions cannot call any wrapper on Julia 1.10 or later, so the compat entry of the package excludes them.

Standard `Base.RefValue` roots usually get their object pointer during copying. Custom conversion roots or methods, and cycles through unfinished boxed callables, defer initialization until the first call. Make that call before concurrent use: lazy initialization is not synchronized. Wrapper copying requires runtime dispatch and is **not trim-safe**.

### PreallocationTools

Loading PreallocationTools.jl enables an extension for `DiffCache` and `FixedSizeDiffCache`. A cache keeps the workspaces that `get_tmp` allocates in an untyped container, `any_du` or `typed_du`, depending on the PreallocationTools version. Copying the entries of that container needs runtime dispatch. So the copy of each container is a new empty container, and `get_tmp` allocates the workspaces of the copy again on demand.

Each container gets one copy, so caches that share a container, such as a cache and its `reshape`, also share it in the copy. If the copy reaches a container through another reference before it reaches a cache that holds it, the cache gets that copy, with its entries. The extension copies all other fields as the generic copier does, so the pointer rule, custom methods and the `TypeError` for a copy of another type apply to them. The trim tests cover copying a `DiffCache` backed by `Vector{Float64}`.

`LazyBufferCache` and `GeneralLazyBufferCache` use the generic field copier. Their buffers are in a `Dict{Any, Any}`, so copying them needs runtime dispatch and is not trim-safe.

### SharedArrays and SparseArrays

Loading SharedArrays enables an extension that copies a `SharedArray` with Base's method. The copy gets a new shared-memory segment on the same workers. A field copy would keep the original's segment, so the workers would write into the original. Base's method copies the bits of the elements. When the element type can hold a raw pointer, each element then gets the rule for array elements: the method for its type, or `PointerCopyError`.

Loading SparseArrays enables an extension for the CHOLMOD solve workspace, which SparseArrays defines from Julia 1.14. The workspace owns C buffers through raw pointers. Base's method makes the copy, a new, empty workspace that allocates its own buffers in a solve.

## Trimmed programs

The generated path targets concrete field and element types. It also splits unions of concrete types and tuples with union elements, up to **16 concrete alternatives**. For example, `Tuple{Int, Union{Missing, Float64}}` has two alternatives. Captured variables in a closure follow the same rules as fields.

Abstract fields such as `Any` require runtime dispatch unless a custom method supplies the concrete type:

```julia
struct Registry
    owner::Any
end
function SpecializedDeepcopy.specialized_deepcopy_internal(x::Registry, copies)
    return Registry(specialized_deepcopy_internal(x.owner::Owner, copies))
end
```

This method requires the stored value to be an `Owner`. A reassigned closure capture lives in a `Core.Box` with an `Any` field and has the same dispatch problem.

The inference hook permits recursive copy calls up to a **type depth of 16**, subject to Julia's hard limits. This counts nested type parameters, not graph depth: `Float64` has depth 1 and `Tuple{Vector{Float64}}` has depth 3. Deeper calls can fail trim compilation. [test/trim/shapes.jl](test/trim/shapes.jl) covers the boundary.

Concrete types alone do not guarantee a trim build: copy methods must also avoid runtime dispatch, and pointer restrictions still apply. Test the intended entry point with `juliac --trim=safe`.

## Other limits and errors

### Shared with Base.deepcopy

`Base.deepcopy` has the same limits:

- Copying is recursive, so a long linked list can overflow the stack. Both implementations raise `StackOverflowError`.
- Objects must satisfy their type's initialization rules. An object with an illegally unset field, made through Julia internals, can crash either implementation.
- A custom method that returns a value of the wrong type raises `TypeError`, at the top level and inside fields and arrays.
- Copying a module raises `ErrorException`.
- Each copied `SharedArray` gets a new segment of its own. A `reshape` of a `SharedArray`, and its `sdata` array, use the segment of the original, but their copies do not use the segment of the copy.
- From Julia 1.11, neither implementation can copy a `GenericMemory` other than `Memory`, such as `AtomicMemory`. This package raises `PointerCopyError`, which names the `ptr` field and asks for a method. `Base.deepcopy` raises `TypeError: in new, expected DataType`.

### Different from Base.deepcopy

In these cases, `Base.deepcopy` crashes or gives a broken copy, and this package tries to fix them:

- If a dictionary key holds a reference to the dictionary, the copied dictionary receives the key before the copy sets the key's fields. When the key's hash reads those fields, `Base.deepcopy` reads them unset. On Julia 1.10 to 1.13 and 1.14-DEV, that crashed the process in our tests. Before the copy sets those fields, two different keys can also compare equal, so that one entry replaces the other. This package keeps the copied entries of each `Dict` whose key type can reach a mutable object, also through a raw pointer. After the copy, it refills a table that lost or misplaced an entry. Each pass checks that the table contains every copied key with its copied value. A refill can change the keys of another table, so the check repeats, with at most one pass more than the number of such tables. If the last pass still refills a table, the copy raises `ArgumentError` and drops no entry. Two copied keys of one such table that compare equal cause this error. Tables whose keys depend on each other in a cycle can also need more passes than the limit. [test/examples/cyclic_key.jl](test/examples/cyclic_key.jl) shows both results.
- From Julia 1.11, `Base.deepcopy` keeps the stale objectids of the original members in the index of the copied `IdSet`, so the copied set does not find most of its members. This package rebuilds the set, so lookups work. On Julia 1.10, both implementations give the same result.

## Performance

Run `julia --project bench/deepcopy.jl`. It reports the lowest batch-average time per copy over one second, plus allocated bytes. Recorded results on Julia 1.13.1, Apple M5 Pro:

| Value | `Base.deepcopy` | `specialized_deepcopy` |
| --- | --- | --- |
| Numeric buffers (10 210 numbers in four vectors) | 0.98 µs, 84.7 kB | 0.80 µs, 84.3 kB |
| Small value (a vector, a struct and a string) | 0.09 µs, 3.7 kB | 0.06 µs, 0.5 kB |
| 10 000 mutable structs in a vector | 658 µs, 1.70 MB | 154 µs, 0.50 MB |
| 100 media (nested structs, dicts and Union fields) | 126 µs, 375 kB | 37 µs, 253 kB |
| Dict of 1 000 vectors, String keys | 123 µs, 391 kB | 77 µs, 301 kB |
| Dict of 1 000 vectors, Symbol keys | 79 µs, 359 kB | 41 µs, 269 kB |
| Dict of 1 000 vectors, mutable keys | 133 µs, 391 kB | 87 µs, 360 kB |
| 1 000 immutable structs holding vectors | 352 µs, 1.11 MB | 128 µs, 761 kB |
| 10 000 mutable structs in a `Vector{Any}` | 675 µs, 1.70 MB | 264 µs, 594 kB |
| Linked list of 1 000 nodes | 71 µs, 122 kB | 47 µs, 122 kB |
| 10 000 `Union{Missing, Float64}` elements | 394 µs, 1.12 MB | 25 µs, 179 kB |
| 1 000 strings | 68 µs, 210 kB | 33 µs, 123 kB |

## Julia internals

Copy construction uses Julia's layout metadata (`T.name.n_uninitialized`, `Base.allocatedinline`, `Base.datatype_pointerfree`) and runtime field setters to preserve undefined and const fields. `Method.recursion_relation` adjusts inference for recursive copy calls, Array copying follows Julia's 1.11+ Memory layout, and Dict copying reads the slot bytes through `Base.isslotfilled`. These internal interfaces are covered by the ordinary and trim tests across supported versions.

## Tests

```sh
julia --project -e 'import Pkg; Pkg.test()'
```

On Julia 1.12+, the suite also builds and runs a trimmed executable. Each trim shape compiles separately so shared compiled code cannot hide an inference failure.

Released Julia versions also run Aqua on the package and JET on the package and extension copy methods. The main suite checks ambiguities with extensions loaded; Aqua's isolated ambiguity check loads only the package. QA uses a separate process and environment in `test/qa/Project.toml`. Nightly skips QA because JET does not support prereleases.

The suite also lists the `Base.deepcopy_internal` methods of Base and of every standard library of the running Julia version. The list must match the expected list for that version, and the package must handle each type in it. A Julia version without an expected list fails this check.

The suite also runs a seeded fuzzer, [test/fuzz.jl](test/fuzz.jl), and explicit cases, [test/fuzz_cases.jl](test/fuzz_cases.jl). For long runs, see the headers of [scripts/fuzz_hammer.jl](scripts/fuzz_hammer.jl) and [scripts/fuzz_trim.jl](scripts/fuzz_trim.jl).

To run the suite on Julia 1.10 through nightly at once, with juliaup installed:

```sh
julia scripts/test_all_versions.jl
```

Each run uses a temporary environment and writes `scripts/logs/tests_<channel>.log`. Any failed channel, including nightly, makes the script fail.

## License

[MIT](LICENSE).

## AI disclosure

AI coding assistants (Claude, by Anthropic) wrote large parts of the code, the tests and the documentation. The maintainer directed the work and reviewed each change before the commit.
