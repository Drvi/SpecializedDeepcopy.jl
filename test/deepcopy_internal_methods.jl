# Included by runtests.jl, which defines run_outcome.

using LinearAlgebra: lu

# A SharedArray element type that holds a raw pointer. The copy does not own the target, so the method
# gives it a null pointer.
struct Address
    pointer::Ptr{Cvoid}
end
SpecializedDeepcopy.specialized_deepcopy_internal(x::Address, copies) = Address(C_NULL)

# Base and the standard libraries define Base.deepcopy_internal for some of their types. The package
# copies each of those types as Base.deepcopy does, with a method of its own or by calling Base's.
@testset "deepcopy_internal methods of Base and the standard libraries" begin
    # The methods of Base and every standard library, by module and type names, for each Julia
    # version. A fresh process loads every standard library, then marks each method whose types the
    # package does not handle. A Julia version without a list fails, so that its methods get a review.
    common = ["Base.GMP: BigInt", "Base.MPFR: BigFloat", "Base: AbstractLock", "Base: Any", "Base: Array",
        "Base: Dict, IdDict", "Base: GenericCondition", "Base: Module", "Base: SimpleVector", "Base: String",
        "Base: Tuple", "SharedArrays: SharedArray"]
    shared_types = "Base: DataType, GlobalRef, Method, MethodInstance, Regex, Symbol, Task, Union, UnionAll"
    memory = ["Base: GenericMemory", "Base: GenericMemoryRef"]
    expected = Dict(
        v"1.10" => [common; shared_types],
        v"1.11" => [common; shared_types; memory],
        v"1.12" => [common; shared_types; memory],
        v"1.13" => [common; shared_types; memory],
        v"1.14" => [common; memory; "Base: CancellationTokenSource," * chopprefix(shared_types, "Base:");
            "Base: WaitEntryN"; "SparseArrays.CHOLMOD: CholmodWS"; "SparseArrays.CHOLMOD: Dense, Factor, Sparse";
            "SparseArrays.UMFPACK: UmfpackLU"],
    )
    code = """
        for name in readdir(Sys.STDLIB)
            startswith(name, ".") || Base.require(Main, Symbol(name))
        end
        using SpecializedDeepcopy
        library(m) = m.module in (Base, Core) || parentmodule(m.module) === Base ||
            (dir = pkgdir(m.module); dir !== nothing && startswith(dir, Sys.STDLIB))
        generic = which(SpecializedDeepcopy.specialized_deepcopy_internal, Tuple{Any, Any})
        # Base's methods for Any and Tuple copy fields, as the generic copy does. A method signature
        # can drop the bounds of the type's parameters, so the type is narrowed to its declaration.
        # Base's method for GenericMemoryRef covers MemoryRef; other kinds of memory are a limit of
        # both implementations.
        function handled(T)
            (T === Any || T === Tuple) && return true
            T = typeintersect(T, Base.unwrap_unionall(T).name.wrapper)
            nameof(Base.unwrap_unionall(T)) === :GenericMemoryRef && (T = MemoryRef)
            return which(SpecializedDeepcopy.specialized_deepcopy_internal, Tuple{T, IdDict{Any, Any}}) !== generic
        end
        for m in methods(Base.deepcopy_internal)
            library(m) || continue
            types = Base.uniontypes(Base.rewrap_unionall(Base.unwrap_unionall(m.sig).parameters[2], m.sig))
            names = join(sort([string(nameof(Base.unwrap_unionall(T))) for T in types]), ", ")
            println(m.module, ": ", names, all(handled, types) ? "" : " (not handled)")
        end
        """
    # Only the test project and the standard libraries, so that no other installed package shadows one.
    variables = merge(ENV, Dict("JULIA_LOAD_PATH" => join(["@", "@stdlib"], Sys.iswindows() ? ';' : ':')))
    delete!(variables, "JULIA_PROJECT")
    output = IOBuffer()
    command = setenv(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code`, variables)
    errors = IOBuffer()
    finished = success(pipeline(command; stdout = output, stderr = errors))
    finished || print(String(take!(errors)))
    inventory = sort(split(strip(String(take!(output))), '\n'))
    @test finished
    @test inventory == sort(get(expected, VersionNumber(VERSION.major, VERSION.minor), String[]))

    # Both implementations share these values.
    shared = Any[:name, current_task(), r"a+", GlobalRef(Base, :sin), first(methods(sin)), Int, Union{Int, Nothing}, Vector]
    @static if isdefined(Core, :CancellationTokenSource)
        push!(shared, Core.CancellationTokenSource())
    end
    for x in shared, copier in (specialized_deepcopy, deepcopy)
        @test copier(x) === x
    end

    # Both implementations copy these values, with new mutable contents.
    vector = [1]
    held = ReentrantLock()
    lock(held)
    condition = Threads.Condition()
    for copier in (specialized_deepcopy, deepcopy)
        @test copier(big(2)^100) == big(2)^100 && copier("text") == "text"
        copied = copier(big"1.5")
        @test copied == big"1.5" && copied isa BigFloat
        copied = copier(Core.svec(vector, 2))
        @test copied[1] == vector && copied[1] !== vector && copied[2] == 2
        copied = copier((vector, 2))
        @test copied[1] == vector && copied[1] !== vector
        copied = copier(Dict(vector => vector))
        @test only(keys(copied)) == vector && only(keys(copied)) !== vector && only(values(copied)) !== vector
        copied = copier(IdDict(:key => vector))
        @test copied[:key] == vector && copied[:key] !== vector
        copied = copier([vector])
        @test copied[1] == vector && copied[1] !== vector
        copied = copier(held)
        @test copied isa ReentrantLock && copied !== held && !islocked(copied)
        copied = copier(condition)
        @test copied isa Threads.Condition && copied !== condition && copied.lock !== condition.lock
        @static if isdefined(Base, :Memory)
            memory = Memory{Vector{Int}}(undef, 1)
            memory[1] = vector
            copied = copier(memory)
            @test copied[1] == vector && copied[1] !== vector
            copied = copier(memoryref(memory, 1))
            @test copied.mem !== memory && copied[] == vector && copied[] !== vector
        end
        @static if isdefined(Core, :WaitEntryN)
            entry = Core.WaitEntryN(nothing, 2)
            copied = copier(entry)
            @test copied isa Core.WaitEntryN && copied !== entry
        end
    end
    unlock(held)

    # A CHOLMOD workspace that holds buffers from a solve. Both implementations give a repeated
    # workspace one new, empty copy, which solves with buffers of its own.
    @static if isdefined(SparseArrays.CHOLMOD, :CholmodWS)
        matrix = sparse([4.0 1.0; 1.0 3.0])
        factor = cholesky(matrix)
        rhs = [1.0, 2.0]
        workspace = SparseArrays.CHOLMOD.CholmodWS(factor)
        solution = zeros(2)
        ldiv!(solution, factor, rhs; workspace)
        buffers(w) = (w.Y[], w.E[])
        @test solution ≈ matrix \ rhs && any(!=(C_NULL), buffers(workspace))
        for copier in (specialized_deepcopy, deepcopy)
            copied, again = copier((workspace, workspace))
            @test copied === again && copied isa typeof(workspace) && copied !== workspace
            @test all(==(C_NULL), buffers(copied))
            fill!(solution, 0.0)
            ldiv!(solution, factor, rhs; workspace = copied)
            @test solution ≈ matrix \ rhs
            @test all(((o, c),) -> o == C_NULL || c != o, zip(buffers(workspace), buffers(copied)))
            @test SparseArrays.CHOLMOD.free!(copied)
        end
        @test SparseArrays.CHOLMOD.free!(workspace)
    end

    # CHOLMOD matrices and factorizations and UMFPACK factorizations own C memory through raw pointers.
    # Where SparseArrays defines Base's method for them, both implementations give a repeated value one
    # copy with C memory of its own, which works after the original is freed. In older versions,
    # Base.deepcopy copies the addresses, and this package raises PointerCopyError.
    matrix = sparse([4.0 1.0; 1.0 3.0])
    rhs = [1.0, 2.0]
    owners = ((() -> cholesky(matrix), SparseArrays.CHOLMOD, F -> F \ rhs ≈ matrix \ rhs),
        (() -> SparseArrays.CHOLMOD.Sparse(matrix), SparseArrays.CHOLMOD, S -> sparse(S) == matrix),
        (() -> SparseArrays.CHOLMOD.Dense(rhs), SparseArrays.CHOLMOD, D -> vec(Matrix(D)) == rhs),
        (() -> lu(matrix), SparseArrays.UMFPACK, F -> F \ rhs ≈ matrix \ rhs))
    for (make, owner, works) in owners
        if which(Base.deepcopy_internal, Tuple{typeof(make()), IdDict}).module === owner
            for copier in (specialized_deepcopy, deepcopy)
                original = make()
                copied, again = copier((original, original))
                @test copied === again && copied !== original
                finalize(original)
                @test works(copied)
            end
        else
            @test_throws PointerCopyError specialized_deepcopy(make())
        end
    end

    # A SharedArray gets a new shared segment, also when several references reach it.
    array = SharedArray{Float64}(4; pids = [1])
    array .= 1.0
    for copier in (specialized_deepcopy, deepcopy)
        copied, again = copier((array, array))
        @test copied === again && copied == array && copied.id != array.id
        copied[1] = 2.0
        @test array[1] == 1.0
    end
    # Elements of a type that can hold a raw pointer get the rule for Array elements: the method for
    # their type, or PointerCopyError. Base.deepcopy copies their bits.
    storage = [2.0]
    GC.@preserve storage begin
        pointers = SharedArray{Ptr{Float64}}(1; pids = [1])
        pointers[1] = pointer(storage)
        @test_throws PointerCopyError(Ptr{Float64}, nothing) specialized_deepcopy(pointers)
    end
    addresses = SharedArray{Address}(2; pids = [1])
    fill!(addresses, Address(Ptr{Cvoid}(UInt(0x1000))))
    copied, again = specialized_deepcopy((addresses, addresses))
    @test copied === again && copied.id != addresses.id
    @test all(a -> a.pointer == C_NULL, copied) && all(a -> a.pointer != C_NULL, addresses)
    # A worker that writes into the copy changes the copy, not the original.
    code = """
        using Distributed
        addprocs(1)
        @everywhere using SharedArrays
        using SpecializedDeepcopy
        original = SharedArray{Float64}(4; pids = [1, 2])
        original .= 0.0
        copied = specialized_deepcopy(original)
        remotecall_fetch(c -> (c[1] = 7.0; nothing), 2, copied)
        print(original[1], " ", copied[1])
        """
    @test run_outcome(code) == (true, "0.0 7.0")
end
