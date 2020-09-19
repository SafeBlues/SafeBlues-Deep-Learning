cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using CSV, Statistics, OrdinaryDiffEq, DiffEqFlux, DiffEqSensitivity,
      Flux, Plots, Optim

datafile = "Model_II_2048_10"
datan = CSV.read("$datafile.csv",header=false)

Itrue_low = Array(datan[1,:])
Isb_low   = mean(Array(datan[2:11,:]),dims=1)

Itrue = Itrue_low[2:end]
Isb = Isb_low[2:end]

policy_times = [7,14,98,126,210,217,366]
policy_amount = [0.2,0.3,0.8,0.0,0.3,0.2,0.0]

ann = FastChain(FastDense(1, 16, tanh),FastDense(16, 16, tanh), FastDense(16, 2, abs))
p = [0.1;1.2;Float64.(initial_params(ann))]

I0  = Itrue[1]*10^5
S0  = 10^5 - I0
R0  = 0
I20 = Isb[1]*0.1*S0
S20 = 0.1*S0 - I20
R20 = 0

function fnn(du,u,p,t)
    i = searchsortedfirst(policy_times,t)
    δ = p[1]
    δk= p[2]


    du[1] =  -0.00004.*ann([policy_amount[i]],p[3:end])[1]*δ*u[1]*u[2]
    du[2] =   0.00004.*ann([policy_amount[i]],p[3:end])[1]*δ*u[1]*u[2] - δk*ann([policy_amount[i]],p[3:end])[2]*u[2]
    du[3] =   δk*ann([policy_amount[i]],p[3:end])[2]*u[2]

    du[4] =  -0.00004.*ann([policy_amount[i]],p[3:end])[1]*u[4]*u[5]
    du[5] =   0.00004.*ann([policy_amount[i]],p[3:end])[1]*u[4]*u[5] - ann([policy_amount[i]],p[3:end])[2]*u[5]
    du[6] =   ann([policy_amount[i]],p[3:end])[2]*u[5]
end

u0 = [S0,I0,R0,S20,I20,R20]
tspan = (0.0,105.0)
probnn = ODEProblem(fnn,u0,tspan,p)

function predict(p)
    Array(concrete_solve(probnn,Tsit5(),u0,p,saveat=1))
end

function loss(p)
  pred = predict(p)
  #=l = 10*sum(abs2,(pred[2,1:81]./10^5 - Itrue[1:81]) ) +
      sum(abs2,(pred[5,1:81]./(0.1*S0) - Isb[1:81])  ) +
     30*sum(abs2,(pred[5,82:126]./(0.1*S0) - Isb[82:126]) )# +
      #50sum(abs2,(pred[2,82:121]./10^5 - Itrue[82:121]) )
=#
l = sum(abs2,(pred[2,1:81]./10^5 - Itrue[1:81]) ) +
    sum(abs2,(pred[5,1:81]./(0.1*S0) - Isb[1:81])  ) +
    50*sum(abs2,(pred[5,82:106]./(0.1*S0) - Isb[82:106]) )# +
  l,pred
end

cb = function (p,l,pred) #callback function to observe training
  display(l)
  # using `remake` to re-create our `prob` with current parameters `p`
  res = scatter(0:105,Itrue[1:106], label = "Itrue")
  res = scatter!(0:105,Isb[1:106], label = "Isafeblues")

  res = plot!(0:105,pred[2,1:106]./10^5,lw=5, label = "NNtrue")
  res = plot!(0:105,pred[5,1:106]./(0.1*S0),lw=5, label = "NN safe blues")
  display(res)
  return false # Tell it to not halt the optimization. If return true, then optimization stops
end

# Display the ODE with the initial parameter values.
cb(p,loss(p)...)

res = DiffEqFlux.sciml_train(loss, p, ADAM(0.001), cb = cb, maxiters=1000)

#res2 = DiffEqFlux.sciml_train(loss, res.minimizer,
                             #BFGS(initial_stepnorm = 0.0001), cb = cb)


probnn2_sb = ODEProblem(fnn,u0,(0.0,105.0),res.minimizer)
sol_sb = solve(probnn2_sb,Tsit5(),saveat=1)

scatter(0:105,Itrue[1:106], title = "Model 2: SafeBlues Universal ODE SIR", xaxis = "Day", yaxis = "Percentage Infected", label = "True infected", color =:red)
scatter!(0:105,Isb[1:106], label = "Safe Blues Infected", color = :blue)

plot!(0:105,sol_sb[2, 1:106]./10^5,lw=3, label = "Predicted: Infected")
plot!(0:105,sol_sb[5, 1:106]./(0.1*S0),lw=3, label = "Predicted: Mean SafeBlues Infected")
plot!([80-0.01,80+0.01],[0.0,0.085],lw=3,color=:black,label="Training Data End")

savefig("Yoni_Model_1.pdf")

using JLD
save("Yoni_RD_SafeBlues_Traj2_Train81_Policy91.jld", "Itrue", Itrue , "Isb", Isb, "sol_sb_2", sol_sb[2, 1:106], "sol_sb_5", sol_sb[5, 1:106], "p", res.minimizer)

#EXTRAPOLATION UNDER DIFFERENT POLICY DECISIONS

D = load("Yoni_RD_SafeBlues_Traj2_Train81_Policy91.jld")
Itrue = D["Itrue"]
Isb = D["Isb"]
sol_sb_2 = D["sol_sb_2"]
sol_sb_5 = D["sol_sb_5"]
p_opt = D["p"]

# HIGH SOCIAL DIST
policy_times_low = [7,14,98,106]
policy_amount_low = [0.2,0.3,1.0,1.0]
function fnn_low(du,u,p,t)
    i = searchsortedfirst(policy_times_low,t)
    δ = p[1]
    δk= p[2]

    du[1] =  -0.00004.*ann([policy_amount_low[i]],p[3:end])[1]*δ*u[1]*u[2]
    du[2] =   0.00004.*ann([policy_amount_low[i]],p[3:end])[1]*δ*u[1]*u[2] - δk*ann([policy_amount_low[i]],p[3:end])[2]*u[2]
    du[3] =   δk*ann([policy_amount_low[i]],p[3:end])[2]*u[2]

    du[4] =  -0.00004.*ann([policy_amount_low[i]],p[3:end])[1]*u[4]*u[5]
    du[5] =   0.00004.*ann([policy_amount_low[i]],p[3:end])[1]*u[4]*u[5] - ann([policy_amount_low[i]],p[3:end])[2]*u[5]
    du[6] =   ann([policy_amount_low[i]],p[3:end])[2]*u[5]
end

#MEDIUM SOCIAL DIST
policy_times_medium = [7,14,98,106]
policy_amount_medium = [0.2,0.3,0.6,0.6]
function fnn_medium(du,u,p,t)
    i = searchsortedfirst(policy_times_medium,t)
    δ = p[1]
    δk= p[2]

    du[1] =  -0.00004.*ann([policy_amount_medium[i]],p[3:end])[1]*δ*u[1]*u[2]
    du[2] =   0.00004.*ann([policy_amount_medium[i]],p[3:end])[1]*δ*u[1]*u[2] - δk*ann([policy_amount_medium[i]],p[3:end])[2]*u[2]
    du[3] =   δk*ann([policy_amount_medium[i]],p[3:end])[2]*u[2]

    du[4] =  -0.00004.*ann([policy_amount_medium[i]],p[3:end])[1]*u[4]*u[5]
    du[5] =   0.00004.*ann([policy_amount_medium[i]],p[3:end])[1]*u[4]*u[5] - ann([policy_amount_medium[i]],p[3:end])[2]*u[5]
    du[6] =   ann([policy_amount_medium[i]],p[3:end])[2]*u[5]
end

#LOW SOCIAL DIST
policy_times_high = [7,14,98,106]
policy_amount_high = [0.2,0.3,0.2,0.2]
function fnn_high(du,u,p,t)
    i = searchsortedfirst(policy_times_high,t)
    δ = p[1]
    δk= p[2]

    du[1] =  -0.00004.*ann([policy_amount_high[i]],p[3:end])[1]*δ*u[1]*u[2]
    du[2] =   0.00004.*ann([policy_amount_high[i]],p[3:end])[1]*δ*u[1]*u[2] - δk*ann([policy_amount_high[i]],p[3:end])[2]*u[2]
    du[3] =   δk*ann([policy_amount_high[i]],p[3:end])[2]*u[2]

    du[4] =  -0.00004.*ann([policy_amount_high[i]],p[3:end])[1]*u[4]*u[5]
    du[5] =   0.00004.*ann([policy_amount_high[i]],p[3:end])[1]*u[4]*u[5] - ann([policy_amount_high[i]],p[3:end])[2]*u[5]
    du[6] =   ann([policy_amount_high[i]],p[3:end])[2]*u[5]
end

probnn2_sb = ODEProblem(fnn,u0,(0,105.0),p_opt)
sol_sb = solve(probnn2_sb,Tsit5(),saveat=1)

ustart = [sol_sb[1, 81],sol_sb[2, 81],sol_sb[3, 81],sol_sb[4, 81],sol_sb[5, 81],sol_sb[6, 81]]

probnn2_sb = ODEProblem(fnn_low,ustart,(81,105.0),p_opt)
sol_sb_low = solve(probnn2_sb,Tsit5(),saveat=1)

probnn2_sb = ODEProblem(fnn_medium,ustart,(81,105.0),p_opt)
sol_sb_medium = solve(probnn2_sb,Tsit5(),saveat=1)

probnn2_sb = ODEProblem(fnn_high,ustart,(81,105.0),p_opt)
sol_sb_high = solve(probnn2_sb,Tsit5(),saveat=1)

scatter(0:105,Itrue[1:106], color = :red, xaxis = "Day", yaxis = "Percentage Infected", label = "True infected: Average")
plot!(81:105,sol_sb_low[2,:]./10^5,lw=3, label = "Predicted: High social distancing", color = :red)
plot!(81:105,sol_sb_medium[2,:]./10^5,lw=3, label = "Predicted: Moderate social distancing", color = :green)
plot!(81:105,sol_sb_high[2,:]./10^5,lw=3, label = "Predicted: Low social distancing", color = :purple)
plot!([81-0.01,81+0.01],[0.0,0.05], ylims = (0, 0.18), lw=3,color=:black,label="Training Data End")
savefig("Yoni_Model_2.pdf")

tspan = (0.0,106.0)
datasize = 107;
t = range(tspan[1],tspan[2],length=datasize)

R_eff_low = zeros(Float64, length(t), 1)
R_eff_0_2 = zeros(Float64, length(t), 1)
R_eff_0_3 = zeros(Float64, length(t), 1)
R_eff_0_4 = zeros(Float64, length(t), 1)
R_eff_0_5 = zeros(Float64, length(t), 1)
R_eff_0_7 = zeros(Float64, length(t), 1)
R_eff_medium = zeros(Float64, length(t), 1)
R_eff_0_9 = zeros(Float64, length(t), 1)
R_eff_high = zeros(Float64, length(t), 1)
R_eff_1 = zeros(Float64, length(t), 1)

policy_amount_low = [0.2,0.3,1.0,1.0]
policy_amount_0_2 = [0.2,0.3,0.9,0.9]
policy_amount_0_3 = [0.2,0.3,0.8,0.8]
policy_amount_0_4 = [0.2,0.3,0.7,0.7]
policy_amount_medium = [0.2,0.3,0.6,0.6]
policy_amount_0_7 = [0.2,0.3,0.5,0.5]
policy_amount_0_5 = [0.2,0.3,0.4,0.4]
policy_amount_0_9 = [0.2,0.3,0.3,0.3]
policy_amount_high = [0.2,0.3,0.2,0.2]
policy_amount_1 = [0.2,0.3,0.1,0.1]

num = zeros(Float64, length(t), 1)
den = zeros(Float64, length(t), 1)

for i = 1:length(R_eff_low)
  ind = searchsortedfirst(policy_times_low,t[i])
  p = p_opt
  δ = p[1]
  δk= p[2]
  num[i] = 4*ann([policy_amount_low[ind]],p[3:end])[1]*δ
  den[i] = δk*ann([policy_amount_low[ind]],p[3:end])[2]
  R_eff_low[i] = num[i]/den[i]

  ind = searchsortedfirst(policy_times_low,t[i])
  p = p_opt
  δ = p[1]
  δk= p[2]
  num[i] = 4*ann([policy_amount_0_2[ind]],p[3:end])[1]*δ
  den[i] = δk*ann([policy_amount_0_2[ind]],p[3:end])[2]
  R_eff_0_2[i] = num[i]/den[i]


  ind = searchsortedfirst(policy_times_low,t[i])
  p = p_opt
  δ = p[1]
  δk= p[2]
  num[i] = 4*ann([policy_amount_0_3[ind]],p[3:end])[1]*δ
  den[i] = δk*ann([policy_amount_0_3[ind]],p[3:end])[2]
  R_eff_0_3[i] = num[i]/den[i]

  ind = searchsortedfirst(policy_times_low,t[i])
  p = p_opt
  δ = p[1]
  δk= p[2]
  num[i] = 4*ann([policy_amount_0_4[ind]],p[3:end])[1]*δ
  den[i] = δk*ann([policy_amount_0_4[ind]],p[3:end])[2]
  R_eff_0_4[i] = num[i]/den[i]


  ind = searchsortedfirst(policy_times_low,t[i])
  p = p_opt
  δ = p[1]
  δk= p[2]
  num[i] = 4*ann([policy_amount_0_5[ind]],p[3:end])[1]*δ
  den[i] = δk*ann([policy_amount_0_5[ind]],p[3:end])[2]
  R_eff_0_5[i] = num[i]/den[i]

  ind = searchsortedfirst(policy_times_low,t[i])
  p = p_opt
  δ = p[1]
  δk= p[2]
  num[i] = 4*ann([policy_amount_0_7[ind]],p[3:end])[1]*δ
  den[i] = δk*ann([policy_amount_0_7[ind]],p[3:end])[2]
  R_eff_0_7[i] = num[i]/den[i]

  ind = searchsortedfirst(policy_times_medium,t[i])
  p = p_opt
  δ = p[1]
  δk= p[2]
  num[i] = 4*ann([policy_amount_medium[ind]],p[3:end])[1]*δ
  den[i] = δk*ann([policy_amount_medium[ind]],p[3:end])[2]
  R_eff_medium[i] = num[i]/den[i]

  ind = searchsortedfirst(policy_times_low,t[i])
  p = p_opt
  δ = p[1]
  δk= p[2]
  num[i] = 4*ann([policy_amount_0_9[ind]],p[3:end])[1]*δ
  den[i] = δk*ann([policy_amount_0_9[ind]],p[3:end])[2]
  R_eff_0_9[i] = num[i]/den[i]

  ind = searchsortedfirst(policy_times_high,t[i])
  p = p_opt
  δ = p[1]
  δk= p[2]
  num[i] = 4*ann([policy_amount_high[ind]],p[3:end])[1]*δ
  den[i] = δk*ann([policy_amount_high[ind]],p[3:end])[2]
  R_eff_high[i] = num[i]/den[i]

  ind = searchsortedfirst(policy_times_low,t[i])
  p = p_opt
  δ = p[1]
  δk= p[2]
  num[i] = 4*ann([policy_amount_1[ind]],p[3:end])[1]*δ
  den[i] = δk*ann([policy_amount_1[ind]],p[3:end])[2]
  R_eff_1[i] = num[i]/den[i]
end

D = load("Yoni_RD_R0_SafeBlues_Traj2.jld")
tn = D["t"]
R_eff = D["R_eff"]

Policy = [1 ;0.9; 0.8; 0.7; 0.6; 0.5; 0.4]
R_model1 = [R_eff_low[93]; R_eff_0_2[93]; R_eff_0_3[93]; R_eff_0_4[93]; R_eff_medium[93]; R_eff_0_7[93]; R_eff_0_5[93]]

using LaTeXStrings
plot(Policy,R_model1, framestyle = :box, ylims = (0, 7), xlims = (0, 1.2),  legend = :topright, xaxis = "Social distancing strength (s)", yaxis = L"R_{\textrm{eff}}(t)", label = "Model3", linewidth = 3, color = :black)

savefig("Yoni_Model_3.pdf")
