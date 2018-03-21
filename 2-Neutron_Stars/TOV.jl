
## Tolkien - Oppenheimer - Vladivostok relativistic pressure equation

using Plots, LaTeXStrings
#gr()   #faster than pgfplots()
pyplot() #xaxis=(0.0,Umax), yaxis=(0.0,1.05) plot!(u[θ.!=0], ϕ[θ.!=0])
fnt = "Source Sans Pro"
default(titlefont=Plots.font(fnt, 16), guidefont=Plots.font(fnt, 16), tickfont=Plots.font(fnt, 12), legendfont=Plots.font(fnt, 12))


# dove fx è la funzione uguale alla derivata di x, ma che dipende da y
function RK4system(t::Float64, fx, fy, x::Float64, y::Float64, h=1e-3)
    k1 = fy(t, x)
    l1 = fx(t, x, y)
    k2 = fy(t+h/2, x+l1*h/2)
    l2 = fx(t+h/2, x+l1*h/2, y+k1*h/2)
    k3 = fy(t+h/2, x+l2*h/2)
    l3 = fx(t+h/2, x+l2*h/2, y+k2*h/2)
    k4 = fy(t+h, x+l3*h)
    l4 = fx(t+h, x+l3*h, y+k3*h)
    x += (l1 + 2l2 + 2l3 + l4)*h/6
    y += (k1 + 2k2 + 2k3 + k4)*h/6
    return [x, y]
end

function solveLaneEmden(u, n, h, gnam=1.0)  # n is the polytropic index, u the radius array
    println("\nStarting the number crunching...\n")
    umax = u[end]
    μ = zeros(ceil(umax/h)) #massa adimensionale
    P = zeros(ceil(umax/h))

    # SI ma in km e masse solari
    # Ms = 1.9885e30
    # G = 6.67408e-20*Ms
    # ħ = 1.0545718e-40/Ms
    # mn = 1.674927351e-27/Ms # neutron mass
    # ρ0 = 0.16e54*mn
    # ρc = ρ0*gnam
    # c = 299792.458

    # Geometriche con masse solari
    Ms, G, c = 1, 1, 1
    @show ħ = 1.0545718e-27*5.0279e-34*(6.7706e-6)^2/2.0296e5
    @show mn = 1.674927351e-24*5.0279e-34 # neutron mass
    @show ρ0 = 0.16e39/(6.7706e-6)^(3)#*mn
    @show ρ0 = 2e17*1.6199e-12
    ρc = ρ0*gnam

    # Differential system
    if n == 1.5
        @show K = (3*π^2)^(1/3)*ħ^2/(5*mn)      # non-relativistic
        @show K = (ħ*c)/(15π^2*mn*c^2)*(3*π^2/(mn*c^2))^(5/3)
        #K = 1
    elseif n == 3.0
        @show K = c*ħ*(3*π^2)^(1/3)/4   # ultrarelativistic
    else error("NANI")
    end

    #f1(r_, P_, m_) = -1*(G*((P_/K)^(-1-1/n) + P_/c^2))*(m_ + 4π*r_^3*P_/c^2)/(r_^2 - 2r_*G*m_/c^2)
    #f2(r_, P_) = 4π * r_^2 * (P_/K)^(-1-1/n)    #SI km
    #f1(r_,P_,m_) = -1*(P_^(-1-1/n) + P_) * (m_ + 4π*r_^3*P_)/(r_^2 - 2r_*m_)
    #f2(r_,P_) = 4π * r_^2 * P_^(-1-1/n)
    f1(r_, P_, m_) = -1*((P_/K)^(-1-1/n) + P_)*(m_ + 4π*r_^3*P_) / (r_^2 - 2r_*m_)
    f2(r_, P_) = 4π * r_^2 * (P_/K)^(-1-1/n)
    #f1(r_, θ_, m_) = -(G*c^2/(Pc*(n+1)*(c^2*r_^2 - 2G*m_*r_)))*(ρc + Pc*θ_/c^2)*(m_ + 4π*r_^3*Pc*θ_^(n+1)/c^2)
    #f2(r_, θ_) = 4π*ρc*θ_^n * r_^2

    # Initial conditions
    @show P[1] = K*ρc^(1/n+1)
    μ[1] = 0
    @show μ[1] = 4π*h^3/3*ρc
    # Evaluate the differential equations until θ reaches 0 (almost)
    i = 1
    u_ = h
    #while P[i] > P[1]*1e-5
        #@show P[i+1], μ[i+1] = RK4system(u_, f1, f2, P[i], μ[i], h)
        #@show in_pascal = P[i+1]*K/1.8063e-38
        # if P[i+1] < 200
        #     h=1e-5
        #     if P[i+1] < 10
        #         h=1e-9
        #     end
        # end
        #u_ += h
        #i+=1
    #end
    return [P, μ, u_] # where u[i] is the adimensional radius found, μ[i] the a-mass
end #./1.8063e-38


## plot θ e ρ inside the star for different ns
function densityProfiles()
    h = 1e-6
    Umax = 5  # maximum adimensional radius considered
    #u = [h/1e8:h/1e8:h/1e4-h/1e8; linspace(h/1e4, Umax, ceil(Umax/h))]
    u = linspace(h, Umax, ceil(Umax/h))
    @show length(u)
    θ, m, raggio= solveLaneEmden(u, 1.5, h)
    @show raggio
    p1 = plot(u[1:5:find(θ)[end]], θ[1:5:find(θ)[end]], xaxis=(L"R [km]"), ylab=L"\theta", label=L"n = 1.5")
    p2 = plot(u[1:5:find(m)[end]], m[1:5:find(m)[end]], xaxis=(L"R [km]"), ylab=L"M", label=L"n = 1.5")
    # θ, m, = solveLaneEmden(u, 3.0, h)
    # plot!(p1, u[1:5:find(θ)[end]], θ[1:5:find(θ)[end]], label=latexstring("n = 3.0"))
    # plot!(p2, u[1:5:find(m)[end]], m[1:5:find(m)[end]], label=latexstring("n = 3.0"))
    Plots.savefig(p1,"theta3_gr.pdf")    #better to use with the pgfplots() backend
    Plots.savefig(p2,"mass3_profile_gr.pdf")
    gui()
end

densityProfiles()


## Find the radius and mass of the star for a range of n (not yet in physical units)
function radiusMass(nn)
    U = Array{Float64}(length(nn))  # raggio adimensionale
    M = Array{Float64}(length(nn))  # massa adimensioanle
    h = 1e-10
    Umax = 0.00001  # maximum adimensional radius
    u = linspace(h, Umax, ceil(Umax/h))
    for i = 1:length(nn)
        θ, m, U[i] = solveLaneEmden(u, nn[i], h)
        M[i] = m[find(m)[end]]
    end
    return [U,M]
end

# h2 = 1e-3
# Umax2 = 100.0  # maximum adimensional radius
# u2 = linspace(h2, Umax2, ceil(Umax2/h2))
# θ2, m2, U2 = solveLaneEmden(h2, 1.5, h2)
#
# U, M = radiusMass([1.5,3.0])
# p4 = scatter([1.5,3.0], U, leg=false, m=(5,0.9,:blue,Plots.stroke(0)), w=5, xaxis=("n"), yaxis=("Adimensional radius"))
# Plots.savefig(p4,"raggi3_gr.pdf")
# p5 = Plots.plot(U[find(M)], M[find(M)], xaxis=("Adimensional radius"), yaxis=("Adimensional mass"), leg=false)
# savefig(p5,"aradiusamass3_gr.pdf")


# ## Conversion to physical units
# function convertToPhysics(umax, mass, n, fattoremisterioso)
#     #G = 6.67408e-11
#     mn = 1.674927351e-27 # neutron mass
#     ρ0 = 0.16e45*mn
#     ρc = ρ0*fattoremisterioso # central density, ρ0 < ρc < 8ρ0
#     # ħ = 1.0545718e-34
#     # c = 299792458
#     # α = sqrt(K*(ρc)^(1+1/n)*(n+1)/(4π*G*(ρc*mn)^2))
#     Rmax = umax
#     Ms = 1.9885e30 # solar mass (kg)
#     @show Mass =  4π*ρc*mass/Ms
#     return [Rmax, Mass]
# end
#
#
# ## Plots of the radii and mass varying ρc (fattoremisterioso), in the relativistic and non-relativistic case
#
# # Physical nonrelativistic radii and mass
# R_nrgr, M_nrgr = Array{Float64}(8), Array{Float64}(8)
# for i=1:8
#     R_nrgr[i], M_nrgr[i] = convertToPhysics(U[1], M[1], 1.5, i)
# end
# p6 = Plots.scatter(R_nrgr, M_nrgr, xaxis=("R [km]"), yaxis=("Mass [suns]"), leg=false)
# savefig(p6,"radiusmass3_nrgr.pdf")
#
# # Physical ultrarelativistic radii and mass
# R_urgr, M_urgr = Array{Float64}(8), Array{Float64}(8)
# for j=1:8
#     R_urgr[j], M_urgr[j] = convertToPhysics(U[end], M[end], 3.0, j)
# end
#
# p8 = Plots.scatter(R_urgr, M_urgr, xaxis=("R [km]"), yaxis=("Mass [suns]"), leg=false)
# savefig(p8, "radiusmass3_urgr.pdf")
# gui()
