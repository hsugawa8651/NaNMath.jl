# Compare NaNMath.sqrt before and after narrowing the intrinsic method to Float16.
#
# "before" is the method from NaNMath 1.1.4 (src/NaNMath.jl:60), copied here.
# "after" is NaNMath.sqrt from this checkout, which carries the change. Both run
# in the same process and are timed alternately, since shared CI machines make
# absolute times unreliable; only ratios within one job are meant to be read.
#
# Usage (from the repository root):
#   julia --project=bench bench/sqrt_bench.jl

using NaNMath, BenchmarkTools, InteractiveUtils, Printf, Random
using Zygote, ForwardDiff

before(x::T) where {T<:Union{Float16, Float32, Float64}} =
    x < T(0) ? T(NaN) : Base.Intrinsics.sqrt_llvm(x)

const TYPES = (Float16, Float32, Float64)
const N = 10_000
const ROUNDS = 5

out = IOBuffer()
say(args...) = (print(out, args..., "\n"); println(args...))

# --- Machine ----------------------------------------------------------------
# Float16 hardware support, asked only with the constants for this architecture:
# test_cpu_feature returns meaningless answers for another architecture's.
const CPUID = Base.BinaryPlatforms.CPUID
fp16_features = Sys.ARCH === :x86_64 ? (:JL_X86_f16c, :JL_X86_avx512fp16) :
                Sys.ARCH === :aarch64 ? (:JL_AArch64_fullfp16,) : ()
fp16 = [replace(string(f), r"^JL_\w+?_" => "") for f in fp16_features
        if isdefined(CPUID, f) && CPUID.test_cpu_feature(getfield(CPUID, f))]
say("## NaNMath.sqrt: intrinsic for Float16 only")
say()
say("| | |")
say("|---|---|")
say("| OS / arch | ", Sys.KERNEL, " / ", Sys.ARCH, " |")
say("| CPU | ", Sys.cpu_info()[1].model, " (`", Sys.CPU_NAME, "`, ", Sys.CPU_THREADS, " threads) |")
say("| Float16 hardware | ", isempty(fp16) ? "none" : join(fp16, ", "), " |")
say("| Julia | ", VERSION, " |")
say("| NaNMath | ", pkgversion(NaNMath), " at ", pkgdir(NaNMath), " |")
say()

# The checkout must carry the change, or "after" is just "before" again.
m = which(NaNMath.sqrt, (Float64,))
patched = occursin("AbstractFloat", string(m.sig))
say("NaNMath.sqrt(::Float64) dispatches to `", m.sig, "` → change present: **", patched, "**")
say()

# --- Values -----------------------------------------------------------------
Random.seed!(1)
say("### Values (bit-identical before/after)")
say()
say("| type | points | identical | negative → |")
say("|---|---|---|---|")
for T in TYPES
    xs = T[0, -0.0, floatmin(T), nextfloat(T(0)), 0.25, 1, 2, 3, floatmax(T), Inf]
    append!(xs, T.(rand(1000) .* 100))
    same = all(x -> reinterpret(Unsigned, before(x)) == reinterpret(Unsigned, NaNMath.sqrt(x)), xs)
    say("| ", T, " | ", length(xs), " | ", same, " | ", NaNMath.sqrt(T(-1)), " |")
end
say()

# --- Timing -----------------------------------------------------------------
say("### Time per element, `map!` over $N elements (minimum of $ROUNDS alternating rounds)")
say()
say("| type | before (ns) | after (ns) | after / before | Base.sqrt (ns) |")
say("|---|---|---|---|---|")
for T in TYPES
    v = T.(rand(N) .* 10); dst = similar(v)
    tb = Inf; ta = Inf; tbase = Inf
    for _ in 1:ROUNDS
        tb = min(tb, @belapsed map!(before, $dst, $v))
        ta = min(ta, @belapsed map!(NaNMath.sqrt, $dst, $v))
        tbase = min(tbase, @belapsed map!(Base.sqrt, $dst, $v))
    end
    ns(t) = t * 1e9 / N
    say(@sprintf("| %s | %.3f | %.3f | %.2f | %.3f |", T, ns(tb), ns(ta), ta / tb, ns(tbase)))
end
say()

# --- Machine code -----------------------------------------------------------
say("### Machine code: square-root and conversion instructions")
say()
say("Float32 is where the change applies; Float16 keeps the intrinsic and should")
say("be identical. Instructions are listed in order, not deduplicated: without")
say("Float16 hardware, one form converts to Float32 twice and the other once.")
say()
key = r"\b(v?sqrt\w*|fsqrt|v?cvt\w*|fcvt|call\w*|bl)\b"i
skip = ('#', ';', '.', '/')
for T in (Float32, Float16), (name, f) in (("before", before), ("after", NaNMath.sqrt))
    s = sprint(io -> code_native(io, f, (T,); debuginfo=:none))
    insns = [replace(strip(split(l, ('#', ';'))[1]), r"\s+" => " ") for l in split(s, '\n')
             if !isempty(strip(l)) && !startswith(strip(l), skip) && occursin(key, l)]
    say("- ", T, " ", name, ": `", join(insns, "` · `"), "`")
end
say()

# --- AD ---------------------------------------------------------------------
say("### Derivative at 0.3")
say()
say("| type | ForwardDiff | Zygote after | Zygote before |")
say("|---|---|---|---|")
tryz(f, x) = try string(only(Zygote.gradient(f, x))) catch e; first(sprint(showerror, e), 45) end
for T in TYPES
    x = T(0.3)
    say("| ", T, " | ", ForwardDiff.derivative(NaNMath.sqrt, x), " | ", tryz(NaNMath.sqrt, x), " | ", tryz(before, x), " |")
end

if haskey(ENV, "GITHUB_STEP_SUMMARY")
    open(io -> write(io, take!(out)), ENV["GITHUB_STEP_SUMMARY"], "a")
end
