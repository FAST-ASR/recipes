### A Pluto.jl notebook ###
# v0.19.8

using Markdown
using InteractiveUtils

# ╔═╡ 2a646269-f409-4e81-81a2-e8213f93281d
rootdir = joinpath(@__DIR__, "..")

# ╔═╡ 9db495b6-e803-11ec-013f-4d3c3cac6e72
begin
	import Pkg
	Pkg.activate(rootdir)
	import TOML
	using MarkovModels
	import MarkovModels: Semirings
	using JLD2
	using HDF5
	using WordErrorRate
	import Base.Iterators: flatten
	import StatsBase: rle
end

# ╔═╡ 86fcb025-4a15-4f4b-a90e-f8fd16f0b231
md"""
# Phone recognizer
"""

# ╔═╡ ecc880a8-e28e-41b0-b669-769120dd9ddd
config = TOML.parsefile(joinpath(rootdir, "./conf/config.toml"))

# ╔═╡ 756f9ea9-dd3d-49d4-a4f2-d6fa61c37225
md"""
## Prepare references
"""

# ╔═╡ 9cf9e931-c8a7-4a6e-bf6a-2a77830ad00d
begin
	testdir = joinpath(config["dataset"]["dir"], config["dataset"]["name"], config["dataset"]["test"])
	textfile = joinpath(testdir, "text")
	lexfile = joinpath(rootdir, config["graphs"]["lexicon"])
end

# ╔═╡ e0f7e2a0-0d0a-49a3-bf16-1e7cd30d5a86
function prepare_refs(lexfile, textfile)
    lexicon = open(lexfile) do f
        lexicon = Dict{String, Vector{String}}()
        for line in readlines(f)
            word, pron... = split(line)
            lexicon[word] = pron
        end
        lexicon
    end

    open(textfile) do f
        refs = Dict{String, Vector{String}}()
        for line in readlines(f)
            uttid, words... = split(line)
            phones = map(words) do w
                if w in keys(lexicon)
                    lexicon[w]
                else
                    w = replace(w, "," => "", ";" => "", ":" => "", "!" => "", "?" => "")
                    lexicon[w]
                end
            end
            refs[uttid] = flatten(phones) |> collect
        end
        refs
    end
end

# ╔═╡ 9374ac27-2ba8-4a16-b661-904eb6e0c1ce
md"""
## Prepare hypothesis
"""

# ╔═╡ 764c5634-3cb7-4376-8551-35c1c63cc4b8
begin
	graphsdir = joinpath(rootdir, config["graphs"]["dir"], config["dataset"]["name"])
	den_fsmfile = joinpath(graphsdir, "denominator_fsm.jld2")

	outdir = joinpath(rootdir, config["experiment"]["dir"], config["dataset"]["name"], config["output"]["dir"])
	outfile = joinpath(outdir, "test.h5")
end

# ╔═╡ 61c4950f-348a-4555-a8f7-f7bea1e2a3f6
function decode(mfsm::MatrixFSM{SR}, in_lhs::AbstractArray) where SR<:TropicalSemiring
    μ = maxstateposteriors(mfsm, in_lhs)
    path = bestpath(mfsm, μ)
    mfsm.labels[path]
end

# ╔═╡ b364be77-b265-42c2-8c9a-2bf7de874698
function decode_dataset(h5file::String, mfsm::MatrixFSM{SR}; kwargs...) where SR<:TropicalSemiring
    h5open(h5file, "r") do f
        decode_dataset(f, mfsm; kwargs...)
    end
end

# ╔═╡ 5ab751f0-c77d-4c80-9a2b-05953054c735
function decode_dataset(output::HDF5.File, mfsm::MatrixFSM{SR}; acwt=1.0) where SR<:TropicalSemiring
    hyps = Dict{String, Vector{String}}()
    for k in keys(output)
        lhs = acwt * Array(output[k])
        hyp = decode(mfsm, lhs)
        hyps[k] = map(first, hyp)
    end
    return hyps
end

# ╔═╡ 5f952c31-0055-4ee6-a337-146f16a7a182
process_hyp(hyp) = begin
    hyp = first(rle(hyp)) # remove repeated phones
    filter(x -> x != "SIL", hyp) # remove SIL
end

# ╔═╡ 5bd198a1-9057-4677-b4e1-038413109f29
function prepare_hyps(h5file::String, graph::String; kwargs...)
    mfsm = jldopen(graph) do f
        convert(MatrixFSM{TropicalSemiring{Float32}}, f["fsm"])
    end
    prepare_hyps(h5file, mfsm)
end

# ╔═╡ dccb0fe2-65d3-4079-aa93-b5123e4bd8ae
function prepare_hyps(h5file::String, mfsm::MatrixFSM; kwargs...)
    hyps = decode_dataset(h5file, mfsm; kwargs...)
    return Dict(uttid => process_hyp(hyp) for (uttid, hyp) in hyps)
end

# ╔═╡ 75826fd0-3f78-4173-b06f-a74765cd88ed
md"""
## Scoring
"""

# ╔═╡ 4b85195c-511b-4a84-85f9-ff3e5fa0129d
function score(refs::Dict{String, T}, hyps::Dict{String, T}) where T<:Vector{String}
    per_stats = Dict()
    for uttid in keys(refs)
        ref = refs[uttid]
        hyp = hyps[uttid]
        per_stats[uttid] = WER(ref, hyp)
    end
    return per_stats
end

# ╔═╡ 26fc627c-5d91-4f90-a5e4-419b5b9ce825
function compute_per(per_stats::Dict)
    S = foldl(values(per_stats); init=0) do a,b a + b.nsub end
    I = foldl(values(per_stats); init=0) do a,b a + b.nins end
    D = foldl(values(per_stats); init=0) do a,b a + b.ndel end
    N = foldl(values(per_stats); init=0) do a,b a + length(b.ref) end
    return round((S+I+D)/N * 100; digits=2), N, S, I, D
end

# ╔═╡ f6865f59-a095-4701-9e3a-0b61eb2037e4
function dump_stats(io::IO, per_stats::Dict)
    for uttid in sort(collect(keys(per_stats)))
        write(io, uttid)
        write(io, "\n")
        write(io, pralign(String, per_stats[uttid]))
        write(io, "--\n")
    end
end

# ╔═╡ 8a12737c-a28c-4228-ab2b-623446494e44
md"""
## Experiments with different language model weight
"""

# ╔═╡ b486a08d-2021-41c0-951a-bcaf0f38cb79
begin
	refs = prepare_refs(lexfile, textfile)

	mfsm = jldopen(den_fsmfile) do f
	    convert(MatrixFSM{TropicalSemiring{Float32}}, f["fsm"])
	end

	let best_per = Inf, best_per_stats = nothing, best_lmwt = nothing
	    for lmwt in 0.8:0.05:1.2
	        hyps = prepare_hyps(outfile, mfsm; acwt=1/lmwt)
	        per_stats = score(refs, hyps)
	        per_result, nref, nsub, nins, ndel = compute_per(per_stats)

			open(joinpath(outdir, "per_$lmwt"), "w") do f
				write(f, "%PER $per_result [ $(nsub+nins+ndel) / $nref, $nins ins, $ndel del, $nsub sub ]\n")
			end
	        if best_per > per_result
	            best_per = per_result
	            best_per_stats = per_stats
	            best_lmwt = lmwt
	        end
	    end

		per_result, nref, nsub, nins, ndel = compute_per(best_per_stats)
		println("%PER $per_result [ $(nsub+nins+ndel) / $nref, $nins ins, $ndel del, $nsub sub ]")

		open(joinpath(outdir, "best_per"), "w") do f
			write(f, "%PER $per_result [ $(nsub+nins+ndel) / $nref, $nins ins, $ndel del, $nsub sub ] $(joinpath(outdir, "per_$best_lmwt"))\n")
		end

		open(joinpath(outdir, "best_per_details.txt"), "w") do f
			dump_stats(f, best_per_stats)
		end
	end
end

# ╔═╡ Cell order:
# ╠═86fcb025-4a15-4f4b-a90e-f8fd16f0b231
# ╠═2a646269-f409-4e81-81a2-e8213f93281d
# ╠═9db495b6-e803-11ec-013f-4d3c3cac6e72
# ╠═ecc880a8-e28e-41b0-b669-769120dd9ddd
# ╠═756f9ea9-dd3d-49d4-a4f2-d6fa61c37225
# ╠═9cf9e931-c8a7-4a6e-bf6a-2a77830ad00d
# ╠═e0f7e2a0-0d0a-49a3-bf16-1e7cd30d5a86
# ╠═9374ac27-2ba8-4a16-b661-904eb6e0c1ce
# ╠═764c5634-3cb7-4376-8551-35c1c63cc4b8
# ╠═61c4950f-348a-4555-a8f7-f7bea1e2a3f6
# ╠═b364be77-b265-42c2-8c9a-2bf7de874698
# ╠═5ab751f0-c77d-4c80-9a2b-05953054c735
# ╠═5f952c31-0055-4ee6-a337-146f16a7a182
# ╠═5bd198a1-9057-4677-b4e1-038413109f29
# ╠═dccb0fe2-65d3-4079-aa93-b5123e4bd8ae
# ╠═75826fd0-3f78-4173-b06f-a74765cd88ed
# ╠═4b85195c-511b-4a84-85f9-ff3e5fa0129d
# ╠═26fc627c-5d91-4f90-a5e4-419b5b9ce825
# ╠═f6865f59-a095-4701-9e3a-0b61eb2037e4
# ╠═8a12737c-a28c-4228-ab2b-623446494e44
# ╠═b486a08d-2021-41c0-951a-bcaf0f38cb79
