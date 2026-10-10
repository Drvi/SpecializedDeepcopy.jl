# Run fuzz.jl and the explicit cases of fuzz_cases.jl in separate worker processes, to isolate crashes
# and to release compiled types between rounds. Check copies, copies of copies, Base.deepcopy and
# wrapper calls across GC. Failed checks report family, round, seed and the seed of the generator.
#
# From the package directory:
#     julia scripts/fuzz_hammer.jl 60                  # run for 60 minutes
#     julia scripts/fuzz_hammer.jl 60 4                # with 4 workers
#     julia scripts/fuzz_hammer.jl 60 4 --interpreted  # with workers that run --compile=min
#     julia scripts/fuzz_hammer.jl reproduce 7 42          # seed 42 of round 7, in each family
#     julia scripts/fuzz_hammer.jl reproduce 7 method      # scan the family `method` of round 7
#     julia scripts/fuzz_hammer.jl reproduce 7 method 42   # seed 42 of that family only
#     julia scripts/fuzz_hammer.jl reproduce 7 cases       # the explicit cases of round 7
#     julia scripts/fuzz_hammer.jl reproduce --run scripts/logs/run-20261009-201500 7 method 42
# --interpreted runs the driver, its workers and reproduce with --compile=min. The interpreter skips
# compiling copy code for each generated type, so it checks about five times as many seeds, but it does
# not run the compiled code of trimmed programs. Reproduce a failure in the mode that found it, with the
# same Julia version and word size: the values generated for a seed depend on the word size.
# Reproduce accepts a seed for all families, a family to scan until failure or Ctrl-C, or a family and
# a seed. A family scan prints each seed before copying. All forms fail for failed checks.
#
# Each run has a directory scripts/logs/run-<date>-<time>, which later runs leave alone:
#     source/        a copy of the package source, which the workers run
#     environment/   the test environment of the run, with its Manifest
#     run.toml       Julia version and commit, machine, word size, compile mode, run seed and settings
#     hammer.log     the summary
#     round-<n>.log  the log of a worker whose round failed, crashed or timed out
#     round-<n>.case the case and stage that the worker ran last
# The checkout can change during a run. Rounds count from 1 in each run. Each run draws a run seed,
# and the run seed and the round number select the layouts of a round, so each run checks new
# layouts. `reproduce --run <directory>` replays a case with the source, environment, compile mode and
# run seed of that run; without `--run`, reproduce uses the run seed 0. Each round runs each family
# for 120 s and then the explicit cases once.
# A watchdog stops a worker that runs longer than its rounds plus an allowance for startup, and reports
# a timeout. A failure that does not reproduce alone can depend on the earlier seeds of its round:
# scan its family to replay them.
# Direct runs use Pkg.test in a temporary environment for test dependencies. The driver waits for
# active workers after its launch deadline. Ctrl-C stops the active workers before the summary.

# The compile flag of a recorded mode. Base.julia_cmd() keeps the flags of the process that starts a
# replay, so a compiled run needs its flag too.
replay_flag(mode) = mode == "interpreted" ? `--compile=min` : `--compile=yes`

# Workers and replays use the environment that they get; other direct invocations create one through
# Pkg.test.
if abspath(PROGRAM_FILE) == (@__FILE__) && (isempty(ARGS) || ARGS[1] ∉ ("worker", "replay"))
    import Pkg
    # A script exits at once on Ctrl-C. With that off, Pkg.test catches Ctrl-C and gives the test
    # process 4 s to stop its workers and print its summary.
    Base.exit_on_sigint(false)
    interpreted = "--interpreted" in ARGS
    args = filter(!=("--interpreted"), ARGS)
    if length(args) >= 3 && args[1] == "reproduce" && args[2] == "--run"
        # Replay with the source, environment and compile mode of a run.
        import TOML
        run_dir = abspath(args[3])
        environment = joinpath(run_dir, "environment")
        Pkg.activate(environment; io = devnull)
        Pkg.instantiate(; io = devnull)
        recorded = TOML.parsefile(joinpath(run_dir, "run.toml"))
        script = joinpath(run_dir, "source", "scripts", "fuzz_hammer.jl")
        replay = `$(Base.julia_cmd()) $(replay_flag(recorded["mode"])) --startup-file=no --project=$environment $script replay $(joinpath(run_dir, "source")) $(recorded["run_seed"]) $(args[4:end])`
        exit(success(pipeline(replay; stdout, stderr)) ? 0 : 1)
    end
    Pkg.activate(mktempdir(); io = devnull)
    if !isempty(args) && args[1] == "reproduce"
        Pkg.develop(path = dirname(@__DIR__); io = devnull)
        test_args = ["hammer"; args]
    else
        # Copy the package source, so that the workers and later replays run this version.
        package = dirname(@__DIR__)
        run_dir = let stamp = Libc.strftime("%Y%m%d-%H%M%S", time()), suffix = interpreted ? "-interpreted" : "", n = 1
            candidate = joinpath(@__DIR__, "logs", "run-$stamp$suffix")
            while ispath(candidate)
                n += 1
                candidate = joinpath(@__DIR__, "logs", "run-$stamp-$n$suffix")
            end
            candidate
        end
        source = mkpath(joinpath(run_dir, "source"))
        foreach(name -> cp(joinpath(package, name), joinpath(source, name)), ("Project.toml", "src", "ext", "test"))
        mkpath(joinpath(source, "scripts"))
        for name in filter(endswith(".jl"), readdir(@__DIR__))
            cp(joinpath(@__DIR__, name), joinpath(source, "scripts", name))
        end
        Pkg.develop(path = source; io = devnull)
        test_args = ["hammer"; "--run"; run_dir; args]
    end
    # Workers get the compile mode of the driver through Base.julia_cmd().
    julia_args = interpreted ? `--compile=min` : ``
    try
        Pkg.test("SpecializedDeepcopy"; test_args, julia_args)
    catch e
        e isa Union{Pkg.Types.PkgError, InterruptException} || rethrow()
        exit(1)
    end
    exit(0)
end

using SpecializedDeepcopy
using Random: Xoshiro
# The test suite includes this file into a module that already imports the suite's Fuzz and FuzzCases.
isdefined(@__MODULE__, :Fuzz) || include(joinpath(dirname(@__DIR__), "test", "fuzz.jl"))
isdefined(@__MODULE__, :FuzzCases) || include(joinpath(dirname(@__DIR__), "test", "fuzz_cases.jl"))

const ROUND_SECONDS = 120
const PROGRESS_EVERY = 25
const FAMILIES = ("plain", "method", "trim")
# Seconds that a worker can take beyond its rounds, for startup, type definitions, compilation and a
# long last case. On Julia 1.12, a worker starts and defines its types in about 7 s and runs the
# explicit cases in 3 to 8 s; the rest is margin.
const ALLOWANCE = 300
# The flag that runs a reproduction in the compile mode of this process.
const MODE = Base.JLOptions().compile_enabled == 3 ? " --interpreted" : ""

function hammer(args)
    if !isempty(args) && args[1] == "worker"
        return worker(parse(Int, args[2]), parse(Float64, args[3]), args[4], parse(UInt, args[5]))
    elseif !isempty(args) && args[1] == "replay"
        # The replay of a run must load the package from the run's source.
        loaded = pkgdir(SpecializedDeepcopy)
        if realpath(loaded) != realpath(args[2])
            println("SpecializedDeepcopy loads from $loaded, not from the run's source $(args[2])")
            return 1
        end
        return reproduce(reproduce_arguments(args[4:end])...; run_seed = parse(UInt, args[3]))
    elseif !isempty(args) && args[1] == "reproduce"
        return reproduce(reproduce_arguments(args[2:end])...)
    elseif length(args) >= 2 && args[1] == "--run"
        return driver(settings(args[3:end])...; run = args[2])
    end
    throw(ArgumentError("start a run with `julia scripts/fuzz_hammer.jl <minutes>`, which copies the source for replays"))
end

# The duration in minutes and the number of workers of a run.
function settings(args)
    minutes = isempty(args) ? 60.0 : parse(Float64, args[1])
    workers = length(args) >= 2 ? parse(Int, args[2]) : max(1, Sys.CPU_THREADS ÷ 2)
    # A run without time or workers would report success without copying anything.
    isfinite(minutes) && minutes > 0 || throw(ArgumentError("the duration must be a positive number of minutes, not $minutes"))
    workers > 0 || throw(ArgumentError("the number of workers must be positive, not $workers"))
    return minutes, workers
end

# The round, then a seed or a family, then optionally a seed.
function reproduce_arguments(args)
    length(args) in (2, 3) || throw(ArgumentError("reproduce needs a round, then a seed or a family, then optionally a seed"))
    round = parse(Int, args[1])
    target = something(tryparse(Int, args[2]), args[2])
    return length(args) == 3 ? (round, target, parse(Int, args[3])) : (round, target)
end

# The seed of the layouts of a round: the run seed and the round number.
layout_seed(run_seed, round) = hash((run_seed, round))

# Plain, custom-method and trim-oriented layouts; this runner does not compile trimmed programs.
# A round defines all three families in this order, also to replay one of them.
function families(layout)
    rng = Xoshiro(layout)
    return [("plain", Fuzz.define_types!(rng, rand(rng, 6:14)), true),
        ("method", Fuzz.define_types!(rng, rand(rng, 6:14); methods = true), true),
        ("trim", Fuzz.define_types!(rng, rand(rng, 6:14); trim = true, methods = true), true)]
end

# The seed of the generator for a seed of a round. Both seeds are native-width hashes: the same run
# seed, round and seed give other values on a machine with another word size.
generator_seed(layout, seed) = hash((layout, seed))

"""
    exercise(family, with_base, layout, seed; record) -> Vector{String}

Return the failed checks of one generated value. `record(stage)` runs before each stage. An exception
ends the case with its stage and backtrace; checks that failed before it stay in the result.
"""
function exercise(family, with_base, layout, seed; record = stage -> nothing)
    reports = String[]
    note(stage, report) = (report === nothing || push!(reports, stage * ": " * report); nothing)
    stage = ""
    enter(name) = (stage = name; record(name))
    try
        enter("generate")
        rng = Xoshiro(generator_seed(layout, seed))
        depth, width = rand(rng, 2:5), rand(rng, 1:5)
        value = Fuzz.generate(rng, family; depth, width)
        enter("copy")
        copied = specialized_deepcopy(value)
        enter("check of the copy")
        note("copy", Fuzz.check(value, copied))
        enter("copy of the copy")
        again = specialized_deepcopy(copied)
        enter("check of the copy of the copy")
        note("copy of the copy", Fuzz.check(copied, again))
        if with_base
            enter("Base.deepcopy")
            based = deepcopy(value)
            enter("check of Base.deepcopy")
            note("Base.deepcopy", Fuzz.check(value, based; bits_fields = false))
        end
        # Check wrapper calls after a full GC. Dropping the input reference does not prove it was
        # collected. Without wrappers, a GC of the young generation still walks the new objects of the
        # case: in a worker with 200 MiB of live objects it takes about 0.3 ms, and a full GC 150 ms.
        enter("wrapper calls after GC")
        wrappers = Any[o for o in Fuzz.mutables(copied) if o isa Fuzz.FunctionWrapper]
        expected = Any[wrapper(1) for wrapper in wrappers]
        value = based = nothing
        GC.gc(!isempty(wrappers))
        for (wrapper, result) in zip(wrappers, expected)
            wrapper(1) == result || push!(reports, "wrapper: $(wrapper(1)) instead of $result")
        end
    catch e
        e isa InterruptException && rethrow()
        push!(reports, "exception in stage $stage: " * sprint(showerror, e, catch_backtrace()))
    end
    return reports
end

# Run the explicit cases of a round, or only case `only`. Call `done(index, name, failed checks)` after
# each case, before the next one starts, and return these triples for all cases run. All cases are
# built, in order, so that each gets the same values.
function run_cases(round, layout; record = line -> nothing, done = (i, name, reports) -> nothing, only = nothing)
    results = Tuple{Int, String, Vector{String}}[]
    cases = FuzzCases.cases(Xoshiro(generator_seed(layout, :cases)))
    only === nothing || only in eachindex(cases) || throw(ArgumentError("round $round has cases 1 to $(length(cases)), not $only"))
    for (i, case) in pairs(cases)
        only === nothing || i == only || continue
        record("round=$round family=cases seed=$i rng=$(repr(generator_seed(layout, :cases))) case=$(case.name)")
        reports = try
            Base.invokelatest(case.run)
        catch e
            e isa InterruptException && rethrow()
            ["exception in case $(case.name): " * sprint(showerror, e, catch_backtrace())]
        end
        push!(results, (i, case.name, reports))
        done(i, case.name, reports)
    end
    return results
end

# A FAIL line with the case, then the rest of the report, such as a backtrace, indented.
function print_failure(io, prefix, report)
    lines = split(report, '\n')
    println(io, "FAIL ", prefix, " ", lines[1])
    foreach(line -> println(io, "    ", line), lines[2:end])
end

function worker(round, seconds, run, run_seed)
    case_io = open(joinpath(run, "round-$round.case"), "w")
    # Overwrite the record of the current case before each stage, so that a crash leaves it behind.
    function record(line)
        seekstart(case_io)
        write(case_io, line)
        truncate(case_io, position(case_io))
        flush(case_io)
    end
    failures = 0
    # Write the failed checks of a case at once: stdout is a file, and a later crash or SIGKILL loses
    # its buffer.
    function publish(prefix, reports)
        for report in reports
            failures += 1
            print_failure(stdout, prefix, report)
        end
        isempty(reports) || flush(stdout)
    end
    # Define the types first; the time budget of each family starts after this preparation.
    layout = layout_seed(run_seed, round)
    prepared = families(layout)
    for (name, family, with_base) in prepared
        println("FAMILY round=$round family=$name layouts=$(length(family))")
        flush(stdout)
        deadline = time() + seconds
        seed = 0
        while time() < deadline
            seed += 1
            prefix = "round=$round family=$name seed=$seed rng=$(repr(generator_seed(layout, seed)))"
            # The generated constructors are newer than this method's world age.
            reports = Base.invokelatest(exercise, family, with_base, layout, seed; record = stage -> record("$prefix stage=$stage"))
            publish(prefix, reports)
            if seed % PROGRESS_EVERY == 0
                println("PROGRESS round=$round family=$name seed=$seed")
                flush(stdout)
            end
        end
        println("COMPLETED round=$round family=$name seeds=$seed")
        flush(stdout)
    end
    println("FAMILY round=$round family=cases")
    results = run_cases(round, layout; record, done = (i, name, reports) ->
        publish("round=$round family=cases seed=$i rng=$(repr(generator_seed(layout, :cases)))", ["$name: $report" for report in reports]))
    println("COMPLETED round=$round family=cases seeds=$(length(results))")
    close(case_io)
    println("DONE round=$round failures=$failures")
    return failures == 0 ? 0 : 1
end

"""
    reproduce(round, target[, seed]; run_seed = 0)

Run seed `target` of round `round` in each family, scan the family `target` until a check fails or
Ctrl-C, or run `seed` of that family only, in the run with seed `run_seed`. The family `cases` runs
the explicit cases of the round. Return 1 for failed checks or Ctrl-C.
"""
function reproduce(round, target, seed = nothing; run_seed = 0)
    layout = layout_seed(run_seed, round)
    target isa Int || target in (FAMILIES..., "cases") || throw(ArgumentError("no family named $target"))
    failed = false
    try
        if target == "cases"
            function publish(i, name, reports)
                for report in reports
                    print_failure(stdout, "round=$round family=cases seed=$i", "$name: $report")
                    failed = true
                end
                isempty(reports) || flush(stdout)
            end
            run_cases(round, layout; only = seed, done = publish)
            return failed ? 1 : 0
        end
        for (name, family, with_base) in families(layout)
            target isa Int || target == name || continue
            seeds = target isa Int ? (target,) : seed !== nothing ? (seed,) : Iterators.countfrom(1)
            for s in seeds
                target isa Int || seed !== nothing || println("seed $s")
                flush(stdout)
                reports = Base.invokelatest(exercise, family, with_base, layout, s)
                for report in reports
                    print_failure(stdout, "round=$round family=$name seed=$s rng=$(repr(generator_seed(layout, s)))", report)
                end
                failed |= !isempty(reports)
                failed && target isa String && seed === nothing && break
            end
        end
    catch e
        e isa InterruptException || rethrow()
        println("interrupted")
        return 1
    end
    return failed ? 1 : 0
end

# Start `command` with both output streams in the file `log`. One handle serves both: two handles to
# one file write at separate offsets, over each other's output.
launch(command, log) = open(io -> run(pipeline(command; stdout = io, stderr = io); wait = false), log, "w")

# A worker that completes prints DONE and exits with status 1 exactly when it printed a FAIL line.
completed(lines, exitcode, termsignal) =
    any(startswith("DONE"), lines) && termsignal == 0 && exitcode == (any(startswith("FAIL"), lines) ? 1 : 0)

"""
    outcome(lines, exitcode, termsignal; timed_out = false, interrupted = false) -> String

`timeout` and `interrupted` for a worker that the driver stopped; `infrastructure` for a worker that
did not start a family; `crash` for an unexpected exit; `failed` for failed checks; otherwise `ok`.
"""
function outcome(lines, exitcode, termsignal; timed_out = false, interrupted = false)
    timed_out && return "timeout"
    interrupted && return "interrupted"
    any(startswith("FAMILY"), lines) || return "infrastructure"
    completed(lines, exitcode, termsignal) || return "crash"
    return any(startswith("FAIL"), lines) ? "failed" : "ok"
end

# A worker log that shows a failure stays, whatever ended the worker.
keep_log(result, lines) = !(result in ("ok", "interrupted")) || any(startswith("FAIL"), lines)

# Seconds since a monotonic start in nanoseconds.
elapsed(start) = (time_ns() - start) / 1e9
overdue(start, limit) = elapsed(start) > limit

# Stop processes, by force if they do not end within 2 s: a Julia process does not always end on
# SIGTERM. After Ctrl-C, Pkg.test gives the driver 4 s to stop its workers and write its summary.
# kill skips a process that has ended.
function stop!(processes)
    foreach(kill, processes)
    timedwait(() -> all(process_exited, processes), 2.0) === :ok || foreach(p -> kill(p, Base.SIGKILL), processes)
    foreach(wait, processes)
    return
end

# Fields of a line of `key=value` words. The first occurrence of a key counts: a message after the
# fields can contain the same words.
function fields(line)
    found = Dict{String, String}()
    for m in eachmatch(r"(\w+)=(\S+)", line)
        get!(found, String(m[1]), String(m[2]))
    end
    return found
end

# Seeds that each family of a worker completed: exact counts from COMPLETED lines, and for a family
# that did not complete, the seeds before the case in the case file.
function completed_seeds(lines, case)
    seeds = Dict{String, Int}()
    for line in filter(startswith("COMPLETED"), lines)
        f = fields(line)
        seeds[f["family"]] = parse(Int, f["seeds"])
    end
    # A family that did not complete: the case in the record did not finish, unless a PROGRESS line
    # confirms it.
    f = fields(case)
    if haskey(f, "family") && !haskey(seeds, f["family"])
        progress = [parse(Int, fields(l)["seed"]) for l in lines if startswith(l, "PROGRESS") && fields(l)["family"] == f["family"]]
        seeds[f["family"]] = max(parse(Int, f["seed"]) - 1, maximum(progress; init = 0))
    end
    return seeds
end

# Write `pairs` as TOML strings and integers.
function write_toml(path, pairs)
    open(path, "w") do io
        for (key, value) in pairs
            println(io, key, " = ", value isa Integer ? string(value) : "\"" * escape_string(string(value), '"') * "\"")
        end
    end
end

function run_settings(minutes, workers, run_seed, status)
    return [
        "status" => status, "command" => "julia scripts/fuzz_hammer.jl $minutes $workers$MODE",
        "mode" => isempty(MODE) ? "compiled" : "interpreted", "julia" => string(VERSION),
        "julia_commit" => Base.GIT_VERSION_INFO.commit, "machine" => Sys.MACHINE, "word_size" => Sys.WORD_SIZE,
        "cpu_threads" => Sys.CPU_THREADS, "minutes" => string(minutes), "workers" => workers,
        "round_seconds" => ROUND_SECONDS, "allowance_seconds" => ALLOWANCE,
        "run_seed" => run_seed, "source" => pkgdir(SpecializedDeepcopy),
        "started" => Libc.strftime("%Y-%m-%d %H:%M:%S", time()),
    ]
end

function driver(minutes, workers; run, seconds = ROUND_SECONDS, allowance = ALLOWANCE)
    mkpath(run)
    # The environment of this test process, with the path of the package source, for replays.
    environment = mkpath(joinpath(run, "environment"))
    project = Base.active_project()
    cp(project, joinpath(environment, "Project.toml"); force = true)
    manifest = joinpath(dirname(project), "Manifest.toml")
    isfile(manifest) && cp(manifest, joinpath(environment, "Manifest.toml"); force = true)
    run_seed = rand(UInt32)
    metadata = run_settings(minutes, workers, run_seed, "running")
    write_toml(joinpath(run, "run.toml"), metadata)
    limit = length(FAMILIES) * seconds + allowance
    return open(joinpath(run, "hammer.log"), "a") do io
        say(line) = (println(io, line); flush(io); println(line); flush(stdout))
        deadline = time() + 60minutes
        started = time_ns()
        next = 1
        seeds = Dict{String, Int}()
        outcomes = Dict{String, Int}()
        failures = 0
        running = Dict{Int, NamedTuple{(:process, :log, :case, :start), Tuple{Base.Process, String, String, UInt64}}}()
        report = time()
        interrupted = false
        # The rounds whose workers the driver stopped on Ctrl-C.
        stopped = Set{Int}()
        say("hammer: $(round(Int, minutes)) minutes with $workers$(isempty(MODE) ? "" : " interpreted") workers, $seconds s per round and family, run seed $run_seed, run directory $run")
        # Ctrl-C reaches the driver only in `sleep`, so it never comes between starting a worker and
        # recording it, or between removing an ended round and counting it. The whole iteration handles
        # an InterruptException, also one that a test schedules into the task.
        Base.disable_sigint() do
            while time() < deadline || !isempty(running)
                try
                    while length(running) < workers && time() < deadline
                        r = next
                        next += 1
                        log, case = joinpath(run, "round-$r.log"), joinpath(run, "round-$r.case")
                        command = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $(@__FILE__) worker $r $seconds $run $run_seed`
                        running[r] = (process = launch(command, log), log, case, start = time_ns())
                    end
                    Base.reenable_sigint(() -> sleep(1))
                    # The watchdog: a worker past its rounds and allowance is stopped and reported.
                    timed_out = [r for (r, w) in running if !process_exited(w.process) && overdue(w.start, limit)]
                    stop!([running[r].process for r in timed_out])
                    for (r, w) in collect(running)
                        process_exited(w.process) || continue
                        delete!(running, r)
                        lines = readlines(w.log)
                        case = isfile(w.case) ? read(w.case, String) : ""
                        result = outcome(lines, w.process.exitcode, w.process.termsignal; timed_out = r in timed_out, interrupted = r in stopped)
                        outcomes[result] = get(outcomes, result, 0) + 1
                        for (family, n) in completed_seeds(lines, case)
                            seeds[family] = get(seeds, family, 0) + n
                        end
                        for line in filter(startswith("FAIL"), lines)
                            failures += 1
                            say(line)
                        end
                        if result in ("crash", "timeout", "infrastructure")
                            failures += 1
                            tail = join(lines[max(1, end - 20):end], "\n")
                            f = fields(case)
                            say("$(uppercase(result)) round=$r, exit code $(w.process.exitcode), last case: $case\n" *
                                "reproduce the round with `reproduce --run $run $r $(get(f, "family", "<family>"))`, worker log $(w.log)\n$tail")
                        end
                        if !keep_log(result, lines)
                            rm(w.log; force = true)
                            rm(w.case; force = true)
                        end
                    end
                    if time() - report > 60
                        report = time()
                        progress = join(("$(strip(read(w.case, String)))" for w in values(running) if isfile(w.case)), "; ")
                        say("hammer: $(next - 1 - length(running)) rounds done, $(sum(values(seeds); init = 0)) cases, $failures failures, $(round(Int, elapsed(started) / 60)) min; running: $progress")
                    end
                catch e
                    e isa InterruptException && !interrupted || rethrow()
                    # Launch no more rounds, and stop the running ones. The loop collects them.
                    interrupted = true
                    deadline = 0.0
                    union!(stopped, keys(running))
                    stop!([w.process for w in values(running)])
                end
            end
        end
        total = sum(values(seeds); init = 0)
        stop = interrupted ? "interrupted after $(round(Int, elapsed(started) / 60)) min, " : ""
        families_line = isempty(seeds) ? "" : " (" * join(("$family $n" for (family, n) in sort!(collect(seeds); by = first)), ", ") * ")"
        say("hammer: $stop$(next - 1) rounds, $total cases$families_line, $failures failures")
        say("hammer: outcomes " * join(("$n $k" for (k, n) in sort!(collect(outcomes); by = first)), ", "))
        failures == 0 || say("reproduce a failure with `julia scripts/fuzz_hammer.jl reproduce --run $run <round> <family> <seed>`; the worker logs of failed rounds are in $run")
        # A run that checked no case is not a successful run.
        ok = failures == 0 && !interrupted && total > 0
        total == 0 && say("hammer: no case completed")
        write_toml(joinpath(run, "run.toml"), [metadata[2:end]; "status" => ok ? "passed" : interrupted ? "interrupted" : "failed";
            "last_round" => next - 1; "cases" => total; "ended" => Libc.strftime("%Y-%m-%d %H:%M:%S", time())])
        return ok ? 0 : 1
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(hammer(ARGS))
end
