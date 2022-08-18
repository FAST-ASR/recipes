### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ e313e3c6-1ba5-11ec-2657-9574ba676d47
begin
	import Pkg
	Pkg.activate("/mnt/matylda3/ikocour/SCMR/WSJ_mix/notebooks/Project.toml")
	using HDF5
	using PlutoUI
	using ProgressBars
	using SpeechFeatures
	using TOML
	using WAV
end

# ╔═╡ d0aeac70-e8b3-4765-9483-e1cbb6233793
md"""
# Extracting MFCC Features
*[Lucas Ondel](https://lucasondel.github.io/), October 2021*

This notebook extracts the MFCC features for a corpus of audio recordings using the [SpeechFeatures](https://github.com/lucasondel/SpeechFeatures.jl) Julia package.
"""

# ╔═╡ 605fd496-1707-4ce8-84f3-dc2dabe13d4e
TableOfContents()

# ╔═╡ 3bf681b3-9949-4a94-824b-b2f7416751bf
md"""
## Setup
"""

# ╔═╡ b248cbdb-93ac-4e40-92ed-1cbd6f0b8f89
md"""
We load the features configuration file. By default, we look for the file  `config.toml` in the root directory (i.e. the directory containing this notebook). Alternatively, when calling this notebook as a julia script, you can specify another configuration file by setting the `SPEECHLAB_FEATURES_CONFIG` environment variable:
```
$ SPEECHLAB_FEATURES_CONFIG=/path/to/file julia ExtractFeatures.jl
```

An example of configuration file can be found [here](https://lucasondel.github.io/resources/SpeechLab/features/v1/config_mfcc).
"""

# ╔═╡ fb1c9103-ad50-4eeb-b142-ff74f55170c3
rootdir = @__DIR__

# ╔═╡ d8c0ac69-6753-4eb2-bf35-d6a3744eb261
configfile = get(ENV, "SPEECHLAB_FEATURES_CONFIG", joinpath(rootdir, "./conf/config.toml"))

# ╔═╡ c16e546d-f83a-4094-bf0c-31457bef0ef5
begin
	@info "Reading configuration from $configfile."
	config = TOML.parsefile(configfile)
end

# ╔═╡ 6eecc2e3-f1bf-4ba2-94ad-a8c9b527f8d5
md"""
## Input

We assume the `wav.scp` and `utt2spk` files are organized as:
```
<dataset.dir>/
|  +-- <dataset.name>/
|  |   +-- <dataset.subsets[1]>/
|  |   |   +-- wav.scp
|  |   |   +-- utt2spk
|  |   +-- <dataset.subsets[1]>/
|  |   |   +-- wav.scp
|  |   |   +-- utt2spk
|  |   +-- ...
```
The keys `<dataset.dir>`, `<dataset.name>` and `<dataset.subsets>` are taken from the configuration file. 

The audio recordings of the corpus are indicated in `scp` file either as a path to a WAV file or as a command writing the WAV data to `stdout`. For the latter case, the line should end with a `|`. Here is an example for both cases:
```
uttid1 /path/to/file.wav
uttid2 cmd [options...] /path/to/file |
```
"""

# ╔═╡ 9d4a54cd-07e6-489b-8335-9f81d3858492
datasetdir = joinpath(config["dataset"]["dir"], config["dataset"]["name"])

# ╔═╡ 7f721d43-f2e6-4330-bc49-96851fee3b3b
config["dataset"]

# ╔═╡ 12a74edb-96a8-48b3-b082-9f4bcc0a51c0
scps = [(subset, joinpath(datasetdir, subset, "wav.scp"))
	    for subset in [config["dataset"]["train"], config["dataset"]["dev"], config["dataset"]["test"], ]]

# ╔═╡ 94f0a7c2-a59d-41c1-be44-dd740f1f6ca9
md"""
## Output

This notebook outputs one [HDF5](https://www.hdfgroup.org/solutions/hdf5) archive for each subset of the corpus. The archives will be stored in:
```
<features.dir>/
|  +-- <dataset.name>/
|  |   +-- <dataset.subsets[1]>/
|  |   |   +-- <features.name>.h5
|  |   +-- <dataset.subsets[2]>/
|  |   |   +-- <features.name>.h5
|  |   +-- ...
```
The keys `<features.dir>`, `<dataset.name>` and `<features.name>` are taken from the configuration file.
"""

# ╔═╡ 8e6d25e4-2cce-4e80-9f8e-915800be51b4
feadir = joinpath(config["features"]["dir"], config["dataset"]["name"])

# ╔═╡ d03c2274-088e-4472-95b0-458027640fa0
md"""
We make sure the output directories exist.
"""

# ╔═╡ 06088106-1915-43ca-8189-64c3723b48cd
for (subset, scp) in scps
	mkpath(joinpath(feadir, subset))
end

# ╔═╡ 229ca6a1-1ef2-458c-88a8-c17208004eb5
md"""
## Extraction

For each record of the `scp` file, we extract the features  and store them in the output archive.
"""

# ╔═╡ a9bddc6f-bfa9-4812-a1ae-03588f9a0e80
loadpath(path) = wavread(path, format="double")

# ╔═╡ 9ff0cc22-d73d-40e3-aea6-ff76bdb3901f
function loadcmd(cmd)
	subcmds = [`$(split(subcmd))` for subcmd in split(cmd[1:end-1], "|")]
	pipe = pipeline(subcmds...)
	wavread(IOBuffer(read(pipe)))
end

# ╔═╡ 8032e227-bead-4370-ae5b-9a845d81abec
load(str) = endswith(str, "|") ? loadcmd(str) : loadpath(str)

# ╔═╡ ff37d23e-5cc8-4a13-bbb7-877f043f50c5
function load_scprecord(line)
	tokens = split(strip(line))
	uttid = tokens[1]
	path_or_cmd = join(tokens[2:end], " ")
	String(uttid), path_or_cmd
end

# ╔═╡ e223060e-743a-4373-96f7-8a32fe331a3d
function extractfeatures(feaconfig, scp, outarchive)
	open(scp, "r") do f
		fbank = nothing
		lines = readlines(f)
		for line in ProgressBar(lines)
			uttid, path_or_cmd = load_scprecord(line)
			channels, srate = load(path_or_cmd)

			@assert size(channels, 2) == 1

			x = channels[:, 1]
			S, fftlen = stft(x; srate)
			if isnothing(fbank)
				fbank = filterbank(feaconfig["numfilters"]; fftlen)
			end
			mS = fbank * abs.(S)
			fea = mfcc(mS; nceps=feaconfig["nceps"],
					   liftering=feaconfig["liftering"])
			if feaconfig["deltaorder"] > 0
				fea = add_deltas(fea; order=feaconfig["deltaorder"])
			end

			outarchive[uttid] = fea
		end
	end
end

# ╔═╡ 78885d6e-f459-4636-9b22-8a7cb63accd0
for (subset, scp) in scps
	@info "Extracting features for \"$subset\"."

	path = joinpath(feadir, subset, config["features"]["name"] * ".h5")
	h5open(path, "w") do f
		extractfeatures(config["features"], scp, f)
	end
end

# ╔═╡ f1865c3b-f37b-4574-a1c8-d1a46da990d5
md"""
## Normalization

Optionally, we normalize the mean and variance  of the features for each speaker.
"""

# ╔═╡ 4ddfa3c5-08ff-47de-afa4-e76808147d11
function load_utt2spk(path)
	utt2spk = Dict()
	open(path, "r") do f
		for line in eachline(f)
			utt, spk = split(line)
			
			# NOTE: if not `String` there will be a key mismatch
			# when using substrings.
			utt2spk[String(utt)] = String(spk)
		end
	end
	utt2spk
end

# ╔═╡ 4e1d537f-6c54-4880-b3e1-1546f0d5dab7
function compute_cmv(utt2spk, feaarchive)
	# Accumulate the stats.
	counts = Dict()
	stats1 = Dict()
	stats2 = Dict()
	for (utt, spk) in ProgressBar(utt2spk)
		fea = read(feaarchive[utt])
		if  spk ∉ keys(stats1)
			stats1[spk] = sum(fea, dims=2)[:,1]
			stats2[spk] = sum(fea .^ 2, dims=2)[:,1]
		else
			stats1[spk] .+= sum(fea, dims=2)[:,1]
			stats2[spk] .+= sum(fea .^ 2, dims=2)[:,1]
		end
		c = get(counts, spk, 0)
		counts[spk] = c + size(fea, 2)
	end 
	
	# Renormalize.
	mean = Dict()
	var = Dict()
	for spk in keys(counts)
		μ = stats1[spk] ./ counts[spk]
		mean[spk] = μ
		var[spk] = (stats2[spk] ./ counts[spk]) .- (μ .^ 2) 
	end 
	return mean, var
end

# ╔═╡ ff92c92c-69a9-4733-acf0-f42237bd4ce0
config["features"]

# ╔═╡ d5a05d60-3110-4c64-b085-fa0d17c6c57e
for (subset, scp) in scps
	@info "Computing cepstral mean and variance for each speaker of $subset."
	
	utt2spk = load_utt2spk(joinpath(datasetdir, subset, "utt2spk"))
	path = joinpath(feadir, subset, config["features"]["name"] * ".h5")
	mean, var = h5open(path, "r") do fin
		compute_cmv(utt2spk, fin)
	end
	
	@info "Renormalizing features of $subset."
	tmp, io = mktemp(feadir)
	close(io) # We close the file and reopen it with HDF5.
	h5open(path, "r") do fin
		h5open(tmp, "w") do fout	
			for utt in ProgressBar(keys(fin))
				X = read(fin[utt])
				if config["features"]["mean_norm"] || config["features"]["var_norm"]
					X .-= mean[utt2spk[utt]]
				end
				if config["features"]["var_norm"]
					X ./= sqrt.(var[utt2spk[utt]])
				end
                fout[utt] = convert(Array{Float32}, X)
			end
		end
	end
	
	# Overwrite the un-normalized features.
	mv(tmp, path, force=true)
end

# ╔═╡ Cell order:
# ╟─d0aeac70-e8b3-4765-9483-e1cbb6233793
# ╠═e313e3c6-1ba5-11ec-2657-9574ba676d47
# ╠═605fd496-1707-4ce8-84f3-dc2dabe13d4e
# ╟─3bf681b3-9949-4a94-824b-b2f7416751bf
# ╟─b248cbdb-93ac-4e40-92ed-1cbd6f0b8f89
# ╠═fb1c9103-ad50-4eeb-b142-ff74f55170c3
# ╠═d8c0ac69-6753-4eb2-bf35-d6a3744eb261
# ╠═c16e546d-f83a-4094-bf0c-31457bef0ef5
# ╟─6eecc2e3-f1bf-4ba2-94ad-a8c9b527f8d5
# ╠═9d4a54cd-07e6-489b-8335-9f81d3858492
# ╠═7f721d43-f2e6-4330-bc49-96851fee3b3b
# ╠═12a74edb-96a8-48b3-b082-9f4bcc0a51c0
# ╟─94f0a7c2-a59d-41c1-be44-dd740f1f6ca9
# ╠═8e6d25e4-2cce-4e80-9f8e-915800be51b4
# ╟─d03c2274-088e-4472-95b0-458027640fa0
# ╠═06088106-1915-43ca-8189-64c3723b48cd
# ╟─229ca6a1-1ef2-458c-88a8-c17208004eb5
# ╠═78885d6e-f459-4636-9b22-8a7cb63accd0
# ╠═e223060e-743a-4373-96f7-8a32fe331a3d
# ╠═a9bddc6f-bfa9-4812-a1ae-03588f9a0e80
# ╠═9ff0cc22-d73d-40e3-aea6-ff76bdb3901f
# ╠═8032e227-bead-4370-ae5b-9a845d81abec
# ╠═ff37d23e-5cc8-4a13-bbb7-877f043f50c5
# ╟─f1865c3b-f37b-4574-a1c8-d1a46da990d5
# ╠═4ddfa3c5-08ff-47de-afa4-e76808147d11
# ╠═4e1d537f-6c54-4880-b3e1-1546f0d5dab7
# ╠═ff92c92c-69a9-4733-acf0-f42237bd4ce0
# ╠═d5a05d60-3110-4c64-b085-fa0d17c6c57e
