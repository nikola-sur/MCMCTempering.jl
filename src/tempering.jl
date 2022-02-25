using AdvancedHMC
using AdvancedMH
using DynamicPPL


struct Joint{Tℓprior, Tℓll} <: Function
    ℓprior      :: Tℓprior
    ℓlikelihood :: Tℓll
end

function (joint::Joint)(θ)
    return joint.ℓprior(θ) .+ joint.ℓlikelihood(θ)
end


struct TemperedJoint{Tℓprior, Tℓll, T<:Real} <: Function
    ℓprior      :: Tℓprior
    ℓlikelihood :: Tℓll
    β           :: T
end

function (tj::TemperedJoint)(θ)
    return tj.ℓprior(θ) .+ (tj.ℓlikelihood(θ) .* tj.β)
end


function make_tempered_model(
    sampler,
    model::DifferentiableDensityModel,
    β::Real
)
    ℓπ_β = TemperedJoint(model.ℓπ.ℓprior, model.ℓπ.ℓlikelihood, β)
    ∂ℓπ∂θ_β = TemperedJoint(model.∂ℓπ∂θ.ℓprior, model.∂ℓπ∂θ.ℓlikelihood, β)
    model_β = DifferentiableDensityModel(ℓπ_β, ∂ℓπ∂θ_β)
    return model_β
end


function make_tempered_loglikelihood(
    model::DifferentiableDensityModel,
    β::Real
)
    function logπ(z)
        return model.ℓπ.ℓlikelihood(z) * β
    end
    return logπ
end

function get_params(trans)
    return trans.z.θ
end

function get_tempered_loglikelihoods_and_params(
    model,
    sampler::AbstractMCMC.AbstractSampler,
    states,
    k::Integer,
    Δ::Vector{Real},
    Δ_state::Vector{<:Integer}
)
    
    logπk = make_tempered_loglikelihood(model, Δ[Δ_state[k]])
    logπkp1 = make_tempered_loglikelihood(model, Δ[Δ_state[k + 1]])
    
    θk = get_params(states[k][1])
    θkp1 = get_params(states[k + 1][1])
    
    return logπk, logπkp1, θk, θkp1
end


function get_tempered_loglikelihoods_and_params(
    model,
    sampler,
    states,
    k::Integer,
    Δ::Vector{Real},
    Δ_state::Vector{<:Integer}
)

    logπk = make_tempered_loglikelihood(model, Δ[Δ_state[k]], sampler, get_vi(states[k][2]))
    logπkp1 = make_tempered_loglikelihood(model, Δ[Δ_state[k + 1]], sampler, get_vi(states[k + 1][2]))
    
    θk = get_params(states[k][2], sampler)
    θkp1 = get_params(states[k + 1][2], sampler)
    
    return logπk, logπkp1, θk, θkp1
end


function make_tempered_loglikelihood(model::Model, β::Real, sampler::DynamicPPL.Sampler, varinfo_init::DynamicPPL.VarInfo)
    
    function logπ(z)
        varinfo = DynamicPPL.VarInfo(varinfo_init, sampler, z)
        model(varinfo)
        return DynamicPPL.getlogp(varinfo) * β
    end

    return logπ
end

get_vi(state) = state.vi
get_vi(vi::DynamicPPL.VarInfo) = vi

get_params(state, sampler::DynamicPPL.Sampler) = get_vi(state)[sampler]