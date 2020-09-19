cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using CSV, Statistics, OrdinaryDiffEq, DiffEqFlux, DiffEqSensitivity,
      Flux, Plots, Optim

      datan = CSV.read("sir_scenario_04_10000_lown.csv")
      Itrue_low = Array(datan[1,:])
      Isb_low   = mean(Array(datan[2:51,:]),dims=1)

      Itrue = Itrue_low
      Isb = Isb_low

      #=
      Itrue = (1/3)*(Itrue_high .+ Itrue_med .+ Itrue_low)
      Isb = (1/3)*(Isb_high .+ Isb_med .+ Isb_low)
      =#

      policy_times_low = [7,14,98,126, 210, 217, 366]
      policy_amount_low = [0.5,0.4,0.3,0.2, 0.4, 0.5, 0.6]


      policy_times = policy_times_low
      policy_amount = policy_amount_low

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
tspan = (0.0,160.0)
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
l = sum(abs2,(pred[2,1:126]./10^5 - Itrue[1:126]) ) +
    sum(abs2,(pred[5,1:126]./(0.1*S0) - Isb[1:126])  ) +
    50*sum(abs2,(pred[5,127:161]./(0.1*S0) - Isb[127:161]) )# +
  l,pred
end

cb = function (p,l,pred) #callback function to observe training
  display(l)
  # using `remake` to re-create our `prob` with current parameters `p`
  res = scatter(0:160,Itrue[1:161], label = "Itrue")
  res = scatter!(0:160,Isb[1:161], label = "Isafeblues")

  res = plot!(0:160,pred[2,1:161]./10^5,lw=5, label = "NNtrue")
  res = plot!(0:160,pred[5,1:161]./(0.1*S0),lw=5, label = "NN safe blues")
  display(res)
  return false # Tell it to not halt the optimization. If return true, then optimization stops
end

# Display the ODE with the initial parameter values.
cb(p,loss(p)...)

res = DiffEqFlux.sciml_train(loss, p, ADAM(0.001), cb = cb, maxiters=1000)

#res2 = DiffEqFlux.sciml_train(loss, res.minimizer,
                             #BFGS(initial_stepnorm = 0.0001), cb = cb)


probnn2_sb = ODEProblem(fnn,u0,(0.0,160.0),res.minimizer)
sol_sb = solve(probnn2_sb,Tsit5(),saveat=1)

scatter(0:160,Itrue[1:161], title = "Model 1: SafeBlues Universal ODE SIR", xaxis = "Day", yaxis = "Percentage Infected", label = "True infected", color = :red)
scatter!(0:160,Isb[1:161], label = "Safe Blues Infected", color =:blue)

plot!(0:160,sol_sb[2, 1:161]./10^5,lw=3, label = "Predicted: Infected")
plot!(0:160,sol_sb[5, 1:161]./(0.1*S0),lw=3, label = "Predicted: Mean SafeBlues Infected")
plot!([126-0.01,126+0.01],[0.0,0.03],lw=3,color=:black,label="Training Data End")

savefig("Marjin_RD_UODE_SB_Traj2_Train81_Policy91.pdf")

using JLD
save("Marjin_RD_SafeBlues_Traj2_Train81_Policy91.jld", "Itrue", Itrue , "Isb", Isb, "sol_sb_2", sol_sb[2, 1:161], "sol_sb_5", sol_sb[5, 1:161], "p", res.minimizer)

#EXTRAPOLATION UNDER DIFFERENT POLICY DECISIONS

D = load("Marjin_RD_SafeBlues_Traj2_Train81_Policy91.jld")
Itrue = D["Itrue"]
Isb = D["Isb"]
sol_sb_2 = D["sol_sb_2"]
sol_sb_5 = D["sol_sb_5"]
p_opt = D["p"]

# HIGH SOCIAL DIST
policy_times_low = [7,14,98,126, 210, 217, 366]
policy_amount_low = [0.5,0.4,0.3,0.2, 0.2, 0.5, 0.6]
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
policy_times_medium = [7,14,98,126, 210, 217, 366]
policy_amount_medium = [0.5,0.4,0.3,0.2, 0.6, 0.5, 0.6]
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
policy_times_high = [7,14,98,126, 210, 217, 366]
policy_amount_high = [0.5,0.4,0.3,0.2, 0.8, 0.5, 0.6]
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

probnn2_sb = ODEProblem(fnn,u0,(0,160.0),p_opt)
sol_sb = solve(probnn2_sb,Tsit5(),saveat=1)

ustart = [sol_sb[1, 126],sol_sb[2, 126],sol_sb[3, 126],sol_sb[4, 126],sol_sb[5, 126],sol_sb[6, 126]]

probnn2_sb = ODEProblem(fnn_low,ustart,(126,160.0),p_opt)
sol_sb_low = solve(probnn2_sb,Tsit5(),saveat=1)

probnn2_sb = ODEProblem(fnn_medium,ustart,(126,160.0),p_opt)
sol_sb_medium = solve(probnn2_sb,Tsit5(),saveat=1)

probnn2_sb = ODEProblem(fnn_high,ustart,(126,160.0),p_opt)
sol_sb_high = solve(probnn2_sb,Tsit5(),saveat=1)

scatter(0:160,Itrue[1:161], color = :red, xaxis = "Day", yaxis = "Percentage Infected", label = "True infected: Average")
plot!(126:160,sol_sb_low[2,:]./10^5,lw=3, label = "Predicted: High social distancing", color = :red)
plot!(126:160,sol_sb_medium[2,:]./10^5,lw=3, label = "Predicted: Moderate social distancing", color = :green)
plot!(126:160,sol_sb_high[2,:]./10^5,lw=3, label = "Predicted: Low social distancing", color = :purple)
plot!([126-0.01,126+0.01],[0.0,0.03], ylims = (0, 0.11), lw=3,color=:black,label="Training Data End")
savefig("Marjin_Social_Dist_Policy_Variation_Train81_policy91n.pdf")

tspan = (0.0,160.0)
datasize = 161;
t = range(tspan[1],tspan[2],length=datasize)

R_eff_0_1 = zeros(Float64, length(t), 1)
R_eff_low = zeros(Float64, length(t), 1)
R_eff_0_3 = zeros(Float64, length(t), 1)
R_eff_0_4 = zeros(Float64, length(t), 1)
R_eff_0_5 = zeros(Float64, length(t), 1)
R_eff_medium = zeros(Float64, length(t), 1)
R_eff_0_7 = zeros(Float64, length(t), 1)
R_eff_high = zeros(Float64, length(t), 1)
R_eff_0_9 = zeros(Float64, length(t), 1)
R_eff_1 = zeros(Float64, length(t), 1)

num = zeros(Float64, length(t), 1)
den = zeros(Float64, length(t), 1)

policy_amount_0_1 = [0.5,0.4,0.3,0.2, 0.1, 0.5, 0.6]
policy_amount_low = [0.5,0.4,0.3,0.2, 0.2, 0.5, 0.6]
policy_amount_0_3 = [0.5,0.4,0.3,0.2, 0.3, 0.5, 0.6]
policy_amount_0_4 = [0.5,0.4,0.3,0.2, 0.4, 0.5, 0.6]
policy_amount_0_5 = [0.5,0.4,0.3,0.2, 0.5, 0.5, 0.6]
policy_amount_medium = [0.5,0.4,0.3,0.2, 0.6, 0.5, 0.6]
policy_amount_0_7 = [0.5,0.4,0.3,0.2, 0.7, 0.5, 0.6]
policy_amount_high = [0.5,0.4,0.3,0.2, 0.8, 0.5, 0.6]
policy_amount_0_9 = [0.5,0.4,0.3,0.2, 0.9, 0.5, 0.6]
policy_amount_1 = [0.5,0.4,0.3,0.2, 1.0, 0.5, 0.6]

for i = 1:length(R_eff_low)
    ind = searchsortedfirst(policy_times_low,t[i])
    p = p_opt
    δ = p[1]
    δk= p[2]
    num[i] = 4*ann([policy_amount_0_1[ind]],p[3:end])[1]*δ
    den[i] = δk*ann([policy_amount_0_1[ind]],p[3:end])[2]
    R_eff_0_1[i] = num[i]/den[i]

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
  num[i] = 4*ann([policy_amount_0_7[ind]],p[3:end])[1]*δ
  den[i] = δk*ann([policy_amount_0_7[ind]],p[3:end])[2]
  R_eff_0_7[i] = num[i]/den[i]

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
  num[i] = 4*ann([policy_amount_0_9[ind]],p[3:end])[1]*δ
  den[i] = δk*ann([policy_amount_0_9[ind]],p[3:end])[2]
  R_eff_0_9[i] = num[i]/den[i]

  ind = searchsortedfirst(policy_times_low,t[i])
  p = p_opt
  δ = p[1]
  δk= p[2]
  num[i] = 4*ann([policy_amount_1[ind]],p[3:end])[1]*δ
  den[i] = δk*ann([policy_amount_1[ind]],p[3:end])[2]
  R_eff_1[i] = num[i]/den[i]
end

D = load("Marjin_RD_R0_SafeBlues_Traj2.jld")
tn = D["t"]
R_eff = D["R_eff"]


using LaTeXStrings

Policy = [0.9; 0.8;0.7; 0.6;0.5; 0.4; 0.3; 0.2; 0.1]

R_model1 = [R_eff_0_1[128]; R_eff_low[128]; R_eff_0_3[128]; R_eff_0_4[128]; R_eff_0_5[128]; R_eff_medium[128]; R_eff_0_7[128]; R_eff_high[128]; R_eff_0_9[128] ]

plot(Policy,R_model1, framestyle = :box, title = "Effect of social distancing on forecasted reproduction number", ylims = (0, 5), linewidth = 3,  xlims = (0, 1),  legend = :topright, xaxis = "Social distancing strength (s)", yaxis = L"R_{\textrm{eff}}(t)", label = "Model1", color = :black)

savefig("Line_All_Policy_Marjinn_R0_Social_Dist_Policy_Variation_Train81_Policy91.pdf")
