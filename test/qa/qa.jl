# Aqua checks the package; JET checks package definitions and representative extension calls.
# runtests.jl uses a separate environment to keep these tools out of the main test dependencies,
# and skips these checks on Julia prereleases.
using SpecializedDeepcopy
using FunctionWrappers: FunctionWrapper
using PreallocationTools: DiffCache
using Test
import Aqua, JET

@testset "Aqua and JET" begin
    Aqua.test_all(SpecializedDeepcopy)
    JET.test_package(SpecializedDeepcopy; target_modules = (SpecializedDeepcopy,))
    # JET 0.9 cannot analyze extensions as packages; check representative calls instead.
    extension = Base.get_extension(SpecializedDeepcopy, :SpecializedDeepcopyFunctionWrappersExt)
    JET.test_call(specialized_deepcopy, (FunctionWrapper{Int, Tuple{Int}}, IdDict{Any, Any}); target_modules = (extension,))
    extension = Base.get_extension(SpecializedDeepcopy, :SpecializedDeepcopyPreallocationToolsExt)
    JET.test_call(specialized_deepcopy, (DiffCache{Vector{Float64}, Vector{Float64}}, IdDict{Any, Any}); target_modules = (extension,))
end
