# Aqua checks the package; JET checks package definitions and representative extension calls.
# runtests.jl uses a separate environment to keep these tools out of the main test dependencies,
# and skips these checks on Julia prereleases.
using SpecializedDeepcopy
using FunctionWrappers: FunctionWrapper
using PreallocationTools: DiffCache
using SharedArrays: SharedArray
using SparseArrays: SparseArrays
using Test
import Aqua, JET

@testset "Aqua and JET" begin
    Aqua.test_all(SpecializedDeepcopy)
    JET.test_package(SpecializedDeepcopy; target_modules = (SpecializedDeepcopy,))
    # JET 0.9 cannot analyze extensions as packages; check representative calls instead.
    extension = Base.get_extension(SpecializedDeepcopy, :SpecializedDeepcopyFunctionWrappersExt)
    JET.test_call(SpecializedDeepcopy.specialized_deepcopy_internal, (FunctionWrapper{Int, Tuple{Int}}, IdDict{Any, Any}); target_modules = (extension,))
    extension = Base.get_extension(SpecializedDeepcopy, :SpecializedDeepcopyPreallocationToolsExt)
    JET.test_call(SpecializedDeepcopy.specialized_deepcopy_internal, (DiffCache{Vector{Float64}, Vector{Float64}}, IdDict{Any, Any}); target_modules = (extension,))
    extension = Base.get_extension(SpecializedDeepcopy, :SpecializedDeepcopySharedArraysExt)
    JET.test_call(SpecializedDeepcopy.specialized_deepcopy_internal, (SharedArray{Float64, 1}, IdDict{Any, Any}); target_modules = (extension,))
    # The SparseArrays extension has a method only where SparseArrays defines CholmodWS.
    extension = Base.get_extension(SpecializedDeepcopy, :SpecializedDeepcopySparseArraysExt)
    @test extension !== nothing
    @static if isdefined(SparseArrays.CHOLMOD, :CholmodWS)
        JET.test_call(SpecializedDeepcopy.specialized_deepcopy_internal, (SparseArrays.CHOLMOD.CholmodWS{Int}, IdDict{Any, Any}); target_modules = (extension,))
    end
end
