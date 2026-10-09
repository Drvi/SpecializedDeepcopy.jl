# SpecializedDeepcopy

`specialized_deepcopy(x)` generates copy code for concrete types so suitable object graphs can be copied in programs built with `juliac --trim=safe`. `Base.deepcopy` uses runtime dispatch that the trim verifier rejects.

The target contract is **`Base.deepcopy` with stricter pointer handling**: preserve types, values, aliases and cycles, with an explicit ownership policy for pointer-backed state. The current implementation also has the non-pointer differences listed below.

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

Do not mutate the input graph during copying.

## Differences from Base.deepcopy

| Value | `Base.deepcopy` | `specialized_deepcopy` |
| --- | --- | --- |
| Raw pointer (`Ptr`, `Cstring`, `Cwstring`, `Core.LLVMPtr`) | Copies the address unchanged | Throws `PointerCopyError` unless an enclosing type's method handles ownership |
| `WeakRef` | Copies its fields | Throws `PointerCopyError`; field copying bypasses GC registration |
| `WeakKeyDict` | Rebuilds the table with copied keys | Throws `PointerCopyError`, because the table holds its keys as `WeakRef`s |
| `FunctionWrapper` | Retains the original object pointer | The extension copies the callable storage and rebuilds the pointer |
| `DiffCache` / `FixedSizeDiffCache` workspaces | Copies existing workspaces | The extension starts with empty workspace containers |
| `IdSet` | Field copying can break member lookups on Julia 1.11–1.13 | Rebuilds the set |
| Custom `Base.deepcopy_internal` method | Calls that method | Requires a separate `specialized_deepcopy` method |
| Optional non-isbits inline field described below | Calls the value's copy method | Copies its bits without a method call |

Copying pointer bits neither copies nor keeps the target alive. A custom method must decide whether to share or duplicate the resource. A union such as `Union{Nothing, Ptr{Cvoid}}` is rejected only when it holds a pointer. A `WeakKeyDict` needs a method on the type that holds it, for example one that starts an empty table, as `ResultCache` in [test/types.jl](test/types.jl) does.

The last row applies to fields that a constructor can leave unset, with a concrete, immutable, inline type containing no references or raw pointers. It applies **even when the field is initialized**. Unset bits can contain invalid Union tags, and there is no definedness flag to distinguish them. For a non-plain type `T`, declaring the field as `Union{T, Nothing}` allows its custom method to run.

## Behavior shared with Base.deepcopy

Structs are copied field by field without calling constructors. Declared types, const and atomic fields, and legitimately undefined fields are preserved. Mutable copies are registered before their fields are copied, allowing cycles.

A **plain value** is an `isbitstype` value containing no raw pointer. It is returned unchanged without calling a custom method.

Special handling covers:

- `Array`, `Memory`, `MemoryRef`: new storage. On Julia 1.11+, shared array storage stays shared within the copy. Julia 1.10 copies each array separately, as Base does.
- `Dict` whose keys hash by value (plain values, `String`, `Symbol`, and tuples, named tuples and unions of these): the slots are copied with the entries, as Base does. A key that holds a raw pointer is not in this group, because its method can give the copy another pointer. Other `Dict`s and `IdDict`: rebuilt tables for copied keys. Keys whose hashes read unfinished copies need special handling; see the limits below.
- `Core.SimpleVector`: copied elements.
- Strings, big numbers and locks: Base's copy methods. Conditions get a copied lock and an empty wait queue.
- Symbols, types, tasks, regular expressions, methods, method instances and `GlobalRef`: shared with the original.
- Modules: rejected, as by Base.

Other identity-based collections may need a method that reinserts copied keys.

## Custom copy methods

Define `SpecializedDeepcopy.specialized_deepcopy(x::YourType, copies)`. The `copies::IdDict{Any,Any}` argument maps originals to copies, some still under construction.

Custom `Base.deepcopy_internal` methods must be ported to this API. This also applies to standard libraries: a field-copied `SharedArray` can leave worker processes writing into the original's shared memory. For pointers in Base containers, such as `Vector{Ptr{Cvoid}}`, define the method on your enclosing type.

1. Return the same type as `x`.
2. Use `specialized_deepcopy(value, copies)` for nested values.
3. For mutable types, return an existing `copies[x]`, or register the new copy before following a reference that could lead back to `x`.
4. Do not read fields of an unfinished copy obtained from `copies`.

For example, deliberately share a lookup table:

```julia
struct LookupTable
    values::Vector{Float64}
end
SpecializedDeepcopy.specialized_deepcopy(x::LookupTable, copies) = x
```

Use the two-argument method; a one-argument method affects only top-level calls.

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

function SpecializedDeepcopy.specialized_deepcopy(x::OwnerIndex, copies)
    return OwnerIndex(specialized_deepcopy(x.owners, copies))
end
```

For owned C memory, see `OwnedBuffer` in [test/types.jl](test/types.jl). Its method allocates independent storage, preserves both owners while copying bytes, and installs a finalizer.

### FunctionWrappers

Loading FunctionWrappers.jl enables copying of callable storage and reconstruction of the wrapper's object pointer. The extension requires FunctionWrappers 1.1.2 or later. Earlier versions cannot call any wrapper on Julia 1.10 or later, so the compat entry of the package excludes them.

Standard `Base.RefValue` roots usually get their object pointer during copying. Custom conversion roots or methods, and cycles through unfinished boxed callables, defer initialization until the first call. Make that call before concurrent use: lazy initialization is not synchronized. Wrapper copying requires runtime dispatch and is **not trim-safe**.

### PreallocationTools

Loading PreallocationTools.jl enables an extension for `DiffCache` and `FixedSizeDiffCache`. A cache keeps the workspaces that `get_tmp` allocates in an untyped container, `any_du` or `typed_du`, depending on the PreallocationTools version. Copying the entries of that container needs runtime dispatch. So the copy of each container is a new empty container, and `get_tmp` allocates the workspaces of the copy again on demand.

Each container gets one copy, so caches that share a container, such as a cache and its `reshape`, also share it in the copy. If the copy reaches a container through another reference before it reaches a cache that holds it, the cache gets that copy, with its entries. The extension copies all other fields as the generic copier does, so the pointer rule and custom methods apply to them. The trim tests cover copying a `DiffCache` backed by `Vector{Float64}`.

`LazyBufferCache` and `GeneralLazyBufferCache` use the generic field copier. Their buffers are in a `Dict{Any, Any}`, so copying them needs runtime dispatch and is not trim-safe.

## Trimmed programs

The generated path targets concrete field and element types. It also splits unions of concrete types and tuples with union elements, up to **16 concrete alternatives**. For example, `Tuple{Int, Union{Missing, Float64}}` has two alternatives. Captured variables in a closure follow the same rules as fields.

Abstract fields such as `Any` require runtime dispatch unless a custom method supplies the concrete type:

```julia
struct Registry
    owner::Any
end
function SpecializedDeepcopy.specialized_deepcopy(x::Registry, copies)
    return Registry(specialized_deepcopy(x.owner::Owner, copies))
end
```

This method requires the stored value to be an `Owner`. A reassigned closure capture lives in a `Core.Box` with an `Any` field and has the same dispatch problem.

The inference hook permits recursive copy calls up to a **type depth of 16**, subject to Julia's hard limits. This counts nested type parameters, not graph depth: `Float64` has depth 1 and `Tuple{Vector{Float64}}` has depth 3. Deeper calls can fail trim compilation. [test/trim/shapes.jl](test/trim/shapes.jl) covers the boundary.

Concrete types alone do not guarantee a trim build: copy methods must also avoid runtime dispatch, and pointer restrictions still apply. Test the intended entry point with `juliac --trim=safe`.

## Other limits and errors

- Copying is recursive; a long linked list can overflow the stack.
- Objects must satisfy their type's initialization rules. Illegally uninitialized fields created through Julia internals can crash the copy.
- A key that refers back to its table can be inserted before its fields finish copying. If its hash reads those fields, the hash can later change and break lookup. Base also encounters unfinished keys. A custom method must copy hash-relevant fields before the table.
- Copied `Dict`, `IdDict` and `IdSet` tables do not preserve aliases to separately copied internal storage. Base preserves that alias for `IdSet`, but can break its member lookups on Julia 1.11–1.13.
- A custom method returning the wrong nested type raises `TypeError`.
- A builtin with hidden contents and no copy method raises `ArgumentError`.
- On Julia 1.11 and later, a `GenericMemory` other than `Memory`, such as `AtomicMemory`, raises `PointerCopyError` for its `ptr` field. `Base.deepcopy` also fails on it.
- Copying a module raises `ErrorException`.

## Performance

Run `julia --project bench/deepcopy.jl`. It reports the lowest batch-average time per copy over one second, plus allocated bytes. Recorded results on Julia 1.13.1, Apple M5 Pro:

| Value | `Base.deepcopy` | `specialized_deepcopy` |
| --- | --- | --- |
| Numeric buffers (10 210 numbers in four vectors) | 1.00 µs, 84.7 kB | 0.81 µs, 84.3 kB |
| Small value (a vector, a struct and a string) | 0.09 µs, 3.7 kB | 0.05 µs, 0.5 kB |
| 10 000 mutable structs in a vector | 688 µs, 1.70 MB | 153 µs, 0.50 MB |
| 100 media (nested structs, dicts and Union fields) | 126 µs, 362 kB | 36 µs, 253 kB |
| Dict of 1 000 vectors, String keys | 121 µs, 391 kB | 76 µs, 301 kB |
| Dict of 1 000 vectors, Symbol keys | 80 µs, 359 kB | 41 µs, 269 kB |
| 1 000 immutable structs holding vectors | 351 µs, 1.08 MB | 121 µs, 761 kB |
| 10 000 mutable structs in a `Vector{Any}` | 690 µs, 1.70 MB | 262 µs, 594 kB |
| Linked list of 1 000 nodes | 73 µs, 122 kB | 47 µs, 122 kB |
| 10 000 `Union{Missing, Float64}` elements | 403 µs, 1.12 MB | 26 µs, 179 kB |
| 1 000 strings | 67 µs, 210 kB | 33 µs, 123 kB |

## Julia internals

Copy construction uses Julia's layout metadata (`T.name.n_uninitialized`, `Base.allocatedinline`, `Base.datatype_pointerfree`) and runtime field setters to preserve undefined and const fields. `Method.recursion_relation` adjusts inference for recursive copy calls, Array copying follows Julia's 1.11+ Memory layout, and Dict copying reads the slot bytes through `Base.isslotfilled`. These internal interfaces are covered by the ordinary and trim tests across supported versions.

## Tests

```sh
julia --project -e 'import Pkg; Pkg.test()'
```

On Julia 1.12+, the suite also builds and runs a trimmed executable. Each trim shape compiles separately so shared compiled code cannot hide an inference failure.

Released Julia versions also run Aqua on the package and JET on the package and extension copy methods. The main suite checks ambiguities with extensions loaded; Aqua's isolated ambiguity check loads only the package. QA uses a separate process and environment in `test/qa/Project.toml`. Nightly skips QA because JET does not support prereleases.

The seeded fuzzer checks types, values, undefined fields, aliases, cycles and mutable isolation, including shared array storage. It also checks `Base.deepcopy`, with matching methods for the custom test types.

For a longer run with worker processes:

```sh
julia scripts/fuzz_hammer.jl 60
```

This launches workers for 60 minutes, then waits for active rounds to finish. It uses `Pkg.test` in a temporary environment. Workers vary graph types and sizes, copy copies, request GC and call copied wrappers. Summaries go to `scripts/logs/hammer.log`; failed worker logs are retained beside it until the next run.

Reproduce with the same Julia version and code:

```sh
julia scripts/fuzz_hammer.jl reproduce <round> <seed>
julia scripts/fuzz_hammer.jl reproduce <round> <family>
```

The first command checks one seed in each family (a group of generated test types). The second scans the named family (`plain`, `method` or `trim`), printing each seed before running it, until a check fails or the process exits.

To run the suite on Julia 1.10 through nightly at once, with juliaup installed:

```sh
julia scripts/test_all_versions.jl
```

Each run uses a temporary environment and writes `scripts/logs/tests_<channel>.log`. Any failed channel, including nightly, makes the script fail.

## License

[MIT](LICENSE).
