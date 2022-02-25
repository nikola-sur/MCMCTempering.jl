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

# Choose parameter dimensionality and initial parameter value
D = 10; initial_θ = rand(D)
n_samples, n_adapts = 2_000, 1_000

# Define the target distribution
ℓprior(θ) = 0
ℓlikelihood(θ) = logpdf(MvNormal(zeros(D), ones(D)), θ)
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
chain = MCMCTempering.sample(model, MCMCTempering.tempered(sampler, 4), n_samples; discard_initial = n_adapts)

samples = map((x) -> chain[x].z.θ, 1:length(chain))
Plots.histogram(map((x) -> samples[x][1], 1:length(samples))) # Doesn't seem to be mixing well at the moment!