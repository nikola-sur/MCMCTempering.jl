# From https://github.com/nikola-sur/MCMCTempering.jl and https://turing.ml/dev/tutorials/00-introduction/ 
# and https://github.com/TuringLang/MCMCTempering.jl/tree/tor/improvements

include("../../src/MCMCTempering.jl")

using Turing
using .MCMCTempering
using AdvancedHMC
using Random
using Distributions
using ForwardDiff
using Plots
using LinearAlgebra
using Random

# Choose parameter dimensionality and initial parameter value
D = 10; initial_θ = rand(D)
n_samples, n_adapts = 20_000, 10_000

# Define the target distribution
ℓprior(θ) = 0
ℓlikelihood(θ) = logpdf(MvNormal(zeros(D), I), θ)
∂ℓprior∂θ(θ) = (ℓprior(θ), ForwardDiff.gradient(ℓprior, θ))
∂ℓlikelihood∂θ(θ) = (ℓlikelihood(θ), ForwardDiff.gradient(ℓlikelihood, θ))
model = DifferentiableDensityModel(
    MCMCTempering.Joint(ℓprior, ℓlikelihood),
    MCMCTempering.Joint(∂ℓprior∂θ, ∂ℓlikelihood∂θ)
)

# Define a Hamiltonian system
metric = DiagEuclideanMetric(D)
hamiltonian = Hamiltonian(metric, model.ℓπ, model.∂ℓπ∂θ)
initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
integrator = Leapfrog(initial_ϵ)
proposal = AdvancedHMC.NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

sampler = AdvancedHMC.HMCSampler(proposal, metric, adaptor)

Random.seed!(3248291)
chain = MCMCTempering.sample(model, MCMCTempering.tempered(sampler, 4), n_samples; discard_initial = n_adapts)
samples = map((x) -> chain[x].z.θ, 1:length(chain))
# samples, stats = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; progress=true)

Plots.histogram(map((x) -> samples[x][1], 1:length(samples))) # Doesn't seem to be mixing well at the moment!