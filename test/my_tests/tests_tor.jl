include("../../src/MCMCTempering.jl")

using .MCMCTempering
using Test
using Distributions
using AdvancedMH
using MCMCChains
using Bijectors
using LinearAlgebra
using AbstractMCMC
using Plots

include("../utils.jl")
include("../compat.jl")

# Model settings ---
d = 2
μ_true = [-1.0, 1.0]
σ_true = [1.0, √(10.0)]

function logdensity(x)
    logpdf(MvNormal(μ_true, Diagonal(σ_true.^2)), x)
end
model = DensityModel(logdensity)


# Sampler settings ---
nsamples = 20_000
swap_every = 2
n_chains = 10
swapstrategy = MCMCTempering.NonReversibleSwap()
inverse_temperatures = collect(range(0.0, 1.0, length = n_chains))
spl_inner = RWMH(MvNormal(zeros(d), 1e-1I)) # Set up our sampler with a joint multivariate Normal proposal
spl = MCMCTempering.tempered(spl_inner, inverse_temperatures, swapstrategy; adapt=false, swap_every=swap_every)


# Start analysis ---
states = []
callback = StateHistoryCallback(states)
samples = AbstractMCMC.sample(model, spl, nsamples; callback=callback, progress=false);

process_to_chain_history_list = map(states) do state
    state.process_to_chain
end
process_to_chain_history = permutedims(reduce(hcat, process_to_chain_history_list), (2, 1))

state = states[end]
chain = AbstractMCMC.bundle_samples(samples, model, spl.sampler, MCMCTempering.state_for_chain(state), MCMCChains.Chains)
chain_thinned = chain[length(chain) ÷ 2 + 1:5swap_every:end]

samples2 = map((x) -> samples[x].params, 1:length(samples))