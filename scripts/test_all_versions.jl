# Run tests concurrently on CHANNELS, installing missing juliaup channels.
# Usage: julia scripts/test_all_versions.jl
# Logs go to scripts/logs/tests_<channel>.log; failures print in full.
# Exit 1 if any channel fails, including nightly.

# `1.14-nightly` is the nightly build of the release branch of Julia 1.14, which has no release yet.
const CHANNELS = ["1.10", "1.11", "1.12", "1.13", "1.14-nightly", "nightly"]
const REPO = dirname(@__DIR__)
const LOGS = mkpath(joinpath(@__DIR__, "logs"))

if Sys.which("juliaup") === nothing
    println(stderr, "juliaup is not on the PATH. Install it from https://github.com/JuliaLang/juliaup.")
    exit(1)
end

# Ignore the `*` marking juliaup's default channel.
installed = Set(first(filter(!=("*"), split(line))) for line in eachline(`juliaup status`) if !isempty(strip(line)))
for channel in CHANNELS
    channel in installed || run(`juliaup add $channel`)
end

# Separate environments prevent concurrent writes to the repository's Manifest.toml.
const TEST = "import Pkg; Pkg.activate(mktempdir()); Pkg.develop(path = ARGS[1]); Pkg.test(\"SpecializedDeepcopy\")"

# File redirection hides terminal color support; pass on the caller's setting.
const COLOR = get(stdout, :color, false) ? "yes" : "no"

stamp() = Libc.strftime("[%H:%M:%S]", time())

println("$(stamp()) Testing on Julia $(join(CHANNELS, ", ")). The logs are in $LOGS.")
failed = String[]
@sync for channel in CHANNELS
    @async begin
        log = joinpath(LOGS, "tests_$(channel).log")
        println("$(stamp()) Julia $channel started.")
        start = time()
        # Passing the filename twice opens it twice with separate offsets, so the streams can overwrite
        # each other. Share one handle instead.
        passed = open(log, "w") do io
            success(pipeline(`julia +$channel --startup-file=no --color=$COLOR -e $TEST $REPO`; stdout = io, stderr = io))
        end
        seconds = round(time() - start; digits = 1)
        if passed
            println("$(stamp()) Julia $channel passed in $seconds s.")
        else
            push!(failed, channel)
            # Keep each failed version's log in one print call.
            print("$(stamp()) ===== Julia $channel failed in $seconds s. Log $log:\n", read(log, String), "===== End of the log for Julia $channel\n")
        end
    end
end
isempty(failed) || (println("$(stamp()) Failed: ", join(failed, ", ")); exit(1))
