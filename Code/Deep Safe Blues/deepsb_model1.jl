#### Setup

cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using CSV, Statistics, OrdinaryDiffEq, DiffEqFlux, DiffEqSensitivity,
      Flux, Plots, Optim, Random

Random.seed!(100)

datafile = "sir_scenario_04_10000_low"
strand_data = CSV.read("$datafile.csv",header=false)
policydatafile = "scenario_04_social_distancing_levels_low"
policy_data = Array(CSV.read("$policydatafile.csv",header=false))[:]
policy_times = 0:365

fulltime = 0:(size(strand_data,2)-1)
training_end = 215

Isb   = Array(strand_data[2:51,1:training_end])
Itrue = Array(strand_data[1,1:training_end])
Isb_extrap   = Array(strand_data[2:51,:])
Itrue_extrap = Array(strand_data[1,:])
Isb_extrap_mean = mean(Isb_extrap,dims=1)



scatter(fulltime,Itrue_extrap,lw=3,label="No. Infected")
plot!(fulltime,Isb_extrap',alpha=0.6,label=false)

#### Deep SafeBlues

ann = Chain(Dense(size(Isb,1), 64, tanh),Dense(64, 64, tanh), Dense(64, 1))
loss() = sum(abs2,ann(Isb)' - Itrue)
cb = function ()
  display(loss())
end
Flux.train!(loss,params(ann),Iterators.repeated((), 2000), ADAM(0.01), cb = cb)

scatter(fulltime,Itrue_extrap,lw=3,label="No. Infected")
plot!(fulltime,ann(Isb_extrap)',lw=5,label="Predicted No. Infected")
plot!([training_end-0.01,training_end+0.01],[0.0,maximum(Itrue_extrap)],lw=3,color=:black,label="Training Data End")
plot!(fulltime,Isb_extrap',alpha=0.6,label=false)

#### UODE

ann2 = FastChain(FastDense(3, 32, tanh),FastDense(32, 32, tanh), FastDense(32, 2, σ))
p2 = Float64.(initial_params(ann2))

function fnn2(du,u,p,t)
    S,I,R = u
    i = searchsortedfirst(policy_times,t)
    z = ann2([S./(0.01*S0),I./(0.0001*S0),policy_data[i]],p)
    k1,β = [1,0.00001].*z

    du[1] = dS = -β*S*I
    du[2] = dI =  β*S*I - k1*I
    du[3] = dR =  k1*I
end

I0  = Itrue_extrap[1]*10^5
S0  = 10^5 - I0
R0  = 0
u0 = [S0,I0,R0]
tspan = (0.0,training_end)
probnn2 = ODEProblem(fnn2,u0,tspan,p2)

function predict2(p)
    Array(concrete_solve(probnn2,Tsit5(),u0,p,saveat=1))
end

function loss2(p)
  pred = predict2(p)
  l = sum(abs2,(pred[2,1:training_end]./10^5 - Itrue_extrap[1:training_end]) )

  l,pred
end

cb2 = function (p,l,pred) #callback function to observe training
  display(l)
  # using `remake` to re-create our `prob` with current parameters `p`
  res = scatter(0:training_end-1,Itrue_extrap[1:training_end])
  res = plot!(0:training_end-1,pred[2,1:training_end]./10^5,lw=5)
  display(res)
  return false # Tell it to not halt the optimization. If return true, then optimization stops
end

# Display the ODE with the initial parameter values.
cb2(p2,loss2(p2)...)

res3 = DiffEqFlux.sciml_train(loss2, p2, ADAM(0.001), cb = cb2, maxiters=1000)
res4 = DiffEqFlux.sciml_train(loss2, res3.minimizer,
                             BFGS(initial_stepnorm = 0.0001), cb = cb2)

probnn2_nosb = ODEProblem(fnn2,u0,(0.0,365),res4.minimizer)
sol_nosb = solve(probnn2_nosb,Tsit5(),saveat=1)

scatter(0:365,Itrue_extrap,title="No SafeBlues Fit", label="Infected")
plot!(0:365,sol_nosb[2,1:366]./10^5,lw=3,label="Predicted Infected")
plot!([training_end-0.01,training_end+0.01],[0.0,maximum(Array(sol_nosb[2,:]./(S0)))],lw=3,color=:black,label="Training Data End")

#### UODE2

ann2 = FastChain(FastDense(1, 32, tanh),FastDense(32, 32, tanh), FastDense(32, 2, σ))
p2 = Float64.(initial_params(ann2))

function fnn2(du,u,p,t)
    S,I,R = u
    i = searchsortedfirst(policy_times,t)
    z = ann2([policy_data[i]],p)
    k1,β = [1,0.00001].*z

    du[1] = dS = -β*S*I
    du[2] = dI =  β*S*I - k1*I
    du[3] = dR =  k1*I
end

I0  = Itrue_extrap[1]*10^5
S0  = 10^5 - I0
R0  = 0
u0 = [S0,I0,R0]
tspan = (0.0,training_end)
probnn2 = ODEProblem(fnn2,u0,tspan,p2)

function predict2(p)
    Array(concrete_solve(probnn2,Tsit5(),u0,p,saveat=1))
end

function loss2(p)
  pred = predict2(p)
  l = sum(abs2,(pred[2,1:training_end]./10^5 - Itrue_extrap[1:training_end]) )

  l,pred
end

cb2 = function (p,l,pred) #callback function to observe training
  display(l)
  # using `remake` to re-create our `prob` with current parameters `p`
  res = scatter(0:training_end-1,Itrue_extrap[1:training_end])
  res = plot!(0:training_end-1,pred[2,1:training_end]./10^5,lw=5)
  display(res)
  return false # Tell it to not halt the optimization. If return true, then optimization stops
end

# Display the ODE with the initial parameter values.
cb2(p2,loss2(p2)...)

res3 = DiffEqFlux.sciml_train(loss2, p2, ADAM(0.001), cb = cb2, maxiters=1000)
res4 = DiffEqFlux.sciml_train(loss2, res3.minimizer,
                             BFGS(initial_stepnorm = 0.0001), cb = cb2)

probnn2_nosb2 = ODEProblem(fnn2,u0,(0.0,365),res4.minimizer)
sol_nosb3 = solve(probnn2_nosb,Tsit5(),saveat=1)

scatter(0:365,Itrue_extrap,title="No SafeBlues Fit", label="Infected")
plot!(0:365,sol_nosb[2,1:366]./10^5,lw=3,label="Predicted Infected")
plot!([training_end-0.01,training_end+0.01],[0.0,maximum(Array(sol_nosb[2,:]./(S0)))],lw=3,color=:black,label="Training Data End")

#### NODE

# Note that this is just for demonstration purposes. It's very clear this won't
# work because an ODE is ill-defined on this data, since the directions are not
# unique in 1 dimension. For a 1D neural ODE to fit this data, you'd need for
# example 0.03 to have either a positive or a negative derivative, which cannot
# be the case since it first goes up and then goes down. This explains why the
# training is unable to find a valid ODE for fitting just the 1D data.

ann3 = FastChain(FastDense(2, 32, tanh),FastDense(32, 32, tanh), FastDense(32, 1))
p3 = Float64.(initial_params(ann3))

function fnn3(u,p,t)
    i = searchsortedfirst(policy_times,t)
    z = ann3(vcat(u,policy_data[i]),p)
end

I0  = Itrue_extrap[1]*10^5
S0  = 10^5 - I0
R0  = 0
u02 = [I0]
tspan = (0.0,training_end)
probnn3 = ODEProblem(fnn3,u0,tspan,p3)

function predict3(p)
    Array(concrete_solve(probnn3,Tsit5(),u0,p,saveat=1))
end

function loss2(p)
  pred = predict3(p)
  l = sum(abs2,(pred[1,1:180]./10^5 - Itrue_extrap[1:180]) )
  l,pred
end

cb2 = function (p,l,pred) #callback function to observe training
  display(l)
  # using `remake` to re-create our `prob` with current parameters `p`
  res = scatter(0:training_end-1,Itrue_extrap[1:training_end])
  res = plot!(0:training_end-1,pred[1,1:training_end]./10^5,lw=5)
  display(res)
  return false # Tell it to not halt the optimization. If return true, then optimization stops
end

# Display the ODE with the initial parameter values.
cb2(p3,loss2(p3)...)

res3 = DiffEqFlux.sciml_train(loss2, p3, ADAM(0.01), cb = cb2, maxiters=100)
probnn2_nosb2 = ODEProblem(fnn3,u02,(0.0,365),res3.minimizer)
sol_nosb3 = solve(probnn2_nosb2,Tsit5(),saveat=1)

scatter(0:365,Itrue_extrap,title="No SafeBlues Fit", label="Infected")
plot!(0:365,sol_nosb3[1,1:366]./10^5,lw=3,label="Predicted Infected")
plot!([training_end-0.01,training_end+0.01],[0.0,maximum(Array(sol_nosb[2,:]./(S0)))],lw=3,color=:black,label="Training Data End")

#### Pure SIR

function f3(du,u,p,t)
    S,I,R = u
    β = p[1]; k = p[2]
    du[1] = dS = -β*S*I
    du[2] = dI =  β*S*I - k*I
    du[3] = dR =  k*I
end

prob2 = ODEProblem(f3,u0,tspan,ones(2))

function predict(θ)
  Array(concrete_solve(prob2,Tsit5(),u0,θ,saveat=1,sensealg=ForwardDiffSensitivity()))
end

function loss(θ)
  pred = predict(exp.(θ))
  l = sum(abs2,pred[2,1:training_end]./(S0) - Itrue_extrap[1:training_end])
  l,pred
end

cb = function (θ,l,pred) #callback function to observe training
  display(string(l,"  ",string(exp.(θ)')))
  # using `remake` to re-create our `prob` with current parameters `p`
  res = plot(0:training_end-1,pred[2,1:training_end]./(S0))
  res = scatter!(0:training_end-1,Itrue_extrap[1:training_end])
  display(res)
  return false # Tell it to not halt the optimization. If return true, then optimization stops
end

p0 = [-12.0,-1]
cb(p0,loss(p0)...)

res3 = DiffEqFlux.sciml_train(loss, p0, ADAM(0.1), cb = cb, maxiters=100)
res4 = DiffEqFlux.sciml_train(loss, res3.minimizer,
                             BFGS(initial_stepnorm = 0.01), cb = cb)

prob3_nosb = ODEProblem(f3,u0,(0.0,365.0),exp.(res4.minimizer))
sol_nosb2 = solve(prob3_nosb,Tsit5(),saveat=1)

scatter(0:365,Itrue_extrap,title="No SafeBlues Fit", label="Infected")
plot!(0:365,sol_nosb2[2,1:366]./10^5,lw=3,label="Predicted Infected")
plot!([training_end-1-0.01,training_end-1+0.01],[0.0,maximum(Array(sol_nosb2[2,:]./(S0)))],lw=3,color=:black,label="Training Data End")

#### Final Plots

scatter(0:365,Itrue_extrap,color=:red,title="SafeBlues vs No SafeBlues 2nd Peak Detection: Model 1", legend = :topleft, ylabel = "Percentage Infected", xlabel="Day",label=" Infected")
plot!(0:365,ann(Isb_extrap)',color=1,lw=4,label="Predicted Infected: SafeBlues")
#plot!(0:365,sol_nosb2[2,1:366]./10^5,lw=3,label="Predicted Infected: SIR No SafeBlues")
plot!(0:365,sol_nosb[2,1:366]./10^5,lw=4,color = 8,label="Predicted Infected: No SafeBlues")
plot!([training_end-0.01,training_end+0.01],[0.0,maximum(Itrue_extrap)],lw=3,color=:black,label="Training Data End")
plot!(0:365,Isb_extrap',alpha=0.25,label=false)

savefig("sbvsnosb_$datafile.png")
savefig("sbvsnosb_$datafile.pdf")

scatter(0:215,Itrue_extrap[1:216],color=:red,title="SafeBlues vs No SafeBlues 2nd Peak Detection: Model I", legend = :topright, ylabel = "Proportion Infected", xlabel="Day",label="Infected")
scatter!(215:235,Itrue_extrap[216:236],color=:orange,label="True Unknown Proportion Infected")
plot!(0:215,ann(Isb_extrap)'[1:216],color=3,lw=4,label="Predicted Infected: Trained on Current Numbers")
plot!(216:235,ann(Isb_extrap)'[217:236],color=1,lw=4,label="Safe Blues Projected Infected")
#plot!(0:365,sol_nosb2[2,1:366]./10^5,lw=3,label="Predicted Infected: SIR No SafeBlues")
#plot!(0:130,sol_nosb[2,1:131]./10^5,lw=4,color = 8,label="Predicted Infected: No SafeBlues")
plot!([training_end-0.01,training_end+0.01],[0.0,maximum(Itrue_extrap)],lw=3,color=:black,label="Most Current Infection Statistics")
plot!(0:235,Isb_extrap'[1:236,:],alpha=0.10,color=:blue,label=false)

savefig("sbvsnosb_short_$datafile.png")
savefig("sbvsnosb_short_$datafile.pdf")
