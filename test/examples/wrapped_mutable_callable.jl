# A FunctionWrapper around a callable with state, as ModelingToolkit stores a parameter declared as
# `f(::Real)`. Base.deepcopy copies the callable, but the copied wrapper keeps the pointer to the
# original callable, so calls on the copy use and change the state of the original. With the
# FunctionWrappers extension, specialized_deepcopy points the copy at its own callable.
#
# Run this file alone from the package directory: julia --project test/examples/wrapped_mutable_callable.jl
# The test suite includes this file and calls test_wrapped_mutable_callable().
if abspath(PROGRAM_FILE) == (@__FILE__) && Base.find_package("FunctionWrappers") === nothing
    # The package lists FunctionWrappers as a weak dependency only, so a direct run adds it to a
    # temporary environment.
    import Pkg
    Pkg.activate(; temp = true, io = devnull)
    Pkg.develop(path = dirname(dirname(@__DIR__)); io = devnull)
    Pkg.add("FunctionWrappers"; io = devnull)
end
using SpecializedDeepcopy
using FunctionWrappers: FunctionWrapper
using Test

# A gain and a count of its calls.
mutable struct CountingGain
    k::Float64
    calls::Int
end
(g::CountingGain)(x) = (g.calls += 1; g.k * x)

function test_wrapped_mutable_callable()
    @testset "wrapped mutable callable" begin
        # Base.deepcopy: a call on the copy counts in the original gain. This runs in a fresh process,
        # because the test suite gives Base.deepcopy a FunctionWrapper method for its fuzz checks.
        code = """
            using FunctionWrappers: FunctionWrapper
            mutable struct CountingGain
                k::Float64
                calls::Int
            end
            (g::CountingGain)(x) = (g.calls += 1; g.k * x)
            gain = CountingGain(2.0, 0)
            copied = deepcopy(FunctionWrapper{Real, Tuple{Real}}(gain))
            copied(1.0)
            print(gain.calls, " ", copied.obj[].calls)
            """
        @test read(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code`, String) == "1 0"

        # specialized_deepcopy: the copy calls its own gain, so the two keep separate state through
        # interleaved calls, also after a garbage collection.
        gain = CountingGain(2.0, 0)
        original = FunctionWrapper{Real, Tuple{Real}}(gain)
        independent = specialized_deepcopy(original)
        copied_gain = independent.obj[]
        @test copied_gain !== gain && independent.ptr != C_NULL
        @test independent.objptr == pointer_from_objref(copied_gain) != original.objptr
        GC.gc()
        for _ in 1:3
            original(1.0)
            independent(1.0)
        end
        @test gain.calls == 3 && copied_gain.calls == 3
        copied_gain.k = 5.0
        @test independent(1.0) == 5.0 && original(1.0) == 2.0
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && test_wrapped_mutable_callable()
