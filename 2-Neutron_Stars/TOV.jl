
## Tolkien - Oppenheimer - Vladivostok relativistic pressure equation

using Plots, LaTeXStrings
#gr()   #faster than pgfplots()
pyplot() #xaxis=(0.0,Umax), yaxis=(0.0,1.05) plot!(u[ρ.!=0], ϕ[ρ.!=0])
fnt = "Source Sans Pro"
default(titlefont=Plots.font(fnt, 16), guidefont=Plots.font(fnt, 16), tickfont=Plots.font(fnt, 12), legendfont=Plots.font(fnt, 12))

function integrateAll(arg, h)
    s1 = sum(arg[3:2:end-2])
    s2 = sum(arg[2:2:end-1])
    return (arg[1] + arg[end] + 2*s1 + 4*s2) * h/3     # Regola di Simpson cubica
end

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
    ρ = zeros(ceil(umax/h))
    g = 1+1/n

    # SI ma in km e masse solari
    Ms = 1.9885e30
    G = 6.67408e-11*Ms
    ħ = 1.0545718e-34/Ms
    mn = 1.674927351e-27/Ms # neutron mass
    ρ0 = 0.16e45#*mn
    ρc = ρ0*gnam
    c = 299792458

    # SI
    # G = 6.67408e-11
    # ħ = 1.0545718e-34
    # mn = 1.674927351e-27 # neutron mass
    # ρ0 = 0.16e45#*mn
    # ρc = ρ0*gnam
    # c = 299792458

    # Geometriche con masse solari
    # Ms, G, c = 1, 1, 1
    # @show ħ = 1.0545718e-27*5.0279e-34*(6.7706e-6)^2/2.0296e5
    # @show mn = 1.674927351e-24*5.0279e-34 # neutron mass
    # ρ0 = 0.16e39/(6.7706e-6)^(3)#*mn
    # #@show ρ0 = 2e17*1.6199e-12
    # @show ρc = ρ0*gnam

    # Differential system
    if n == 1.5
        @show K = (3*π^2)^(2/3)*ħ^2/(5*mn)      # non-relativistic
        #@show K = (ħ*c)/(15π^2*mn*c^2)*(3*π^2/(mn*c^2))^(5/3)
        #K = 1
    elseif n == 3.0
        @show K = c*ħ*(3*π^2)^(1/3)/4   # ultrarelativistic
    else error("NANI")
    end
    Pc = K*ρc^(1/n+1)
    f1(r_, ρ_, m_) = -1*(G*(ρ_*mn + K*ρ_^g/c^2))*(m_ + 4π*r_^3*K*ρ_^g/c^2) / (K*g*ρ_^(1/n)*(r_^2 - 2r_*G*m_/c^2))
    f2(r_, ρ_) = 4π * r_^2 * ρ_ *mn    # SI in masse solari

    # Initial conditions

    @show ρ[1] = ρc
    @show μ[1] = ρc*4π*h^2*mn

    # Evaluate the differential equations until ρ reaches 0 (almost)
    i = 1
    while ρ[i] > ρ[1]*1e-6
        ρ[i+1], μ[i+1] = RK4system(u[i], f1, f2, ρ[i], μ[i], h)
        i+=1
    end
    return [ρ, μ, u[i], μ[i]] # where u[i] is the radius found, μ[i] the mass
end


## plot ρ e ρ inside the star for different ns
function densityProfiles()
    U = zeros(2)  # raggio
    M = zeros(2)  # massa
    h = 0.1
    Umax = 90000.0  # maximum radius considered
    u = linspace(h, Umax, ceil(Umax/h))
    Ms = 1.9885e30
    mn = 1.674927351e-27 # neutron mass

    ρ, m, U[1], M[1] = solveLaneEmden(u, 3.0, h)
    p1 = plot(u[1:5:find(ρ)[end]]./1e3, ρ[1:5:find(ρ)[end]].*mn, xaxis=("R [km]"), yaxis=(L"\rho\ [kg/m^3]"), label=L"n = 3.0", yformatter = :scientific)
    #yaxis=(L"\rho\ \ [10^{-14}\ Suns/m^3]")
    #yaxis(L"\rho", xticks=([0.0 0.3 0.6 0.9 1.2 1.5].*1e-13, ("0" "0.3" "0.6" "0.9" "1.2" "1.5"))

    p2 = plot(u[1:5:find(m)[end]]./1e3, m[1:5:find(m)[end]], xaxis=("R [km]",(0,80.0)), yaxis=("M [suns]",(0,3.5)), label=L"n = 3.0")

    ρ, m, U[2], M[2] = solveLaneEmden(u, 1.5, h)
    plot!(p1, u[1:5:find(ρ)[end]]./1e3, ρ[1:5:find(ρ)[end]].*mn, label=latexstring("n = 1.5"))
    plot!(p2, u[1:5:find(m)[end]]./1e3, m[1:5:find(m)[end]], label=latexstring("n = 1.5"))
    Plots.savefig(p1,"theta3_gr.pdf")
    Plots.savefig(p2,"mass3_profile_gr.pdf")
    gui()
    return [U,M]
end

U, M = densityProfiles()



## Plots of the radii and mass varying ρc (gnam), in the relativistic and non-relativistic case

# Physical nonrelativistic radii and mass
R_nrgr, M_nrgr = Array{Float64}(8), Array{Float64}(8)
for i=1:8
    h = 0.1
    Umax = 30000.0  # maximum radius considered
    u = linspace(h, Umax, ceil(Umax/h))
    _, __, R_nrgr[i], M_nrgr[i] = solveLaneEmden(u, 1.5, h, i)
end
p3 = Plots.scatter(R_nrgr./1e3, M_nrgr, xaxis=("R [km]"), yaxis=("Mass [suns]"), leg=false, m=(7,0.7,:blue,Plots.stroke(0)))
savefig(p3,"radiusmass3_nrgr.pdf")

# Physical ultrarelativistic radii and mass
R_urgr, M_urgr = Array{Float64}(8), Array{Float64}(8)
for j=1:8
    h = 0.1
    Umax = 90000.0  # maximum radius considered
    u = linspace(h, Umax, ceil(Umax/h))
    _, __, R_urgr[j], M_urgr[j] = solveLaneEmden(u, 3.0, h, j)
end

p4 = Plots.scatter(R_urgr./1e3, M_urgr, xaxis=("R [km]"), yaxis=("Mass [suns]"), leg=false,m=(7,0.7,:blue,Plots.stroke(0)))
savefig(p4, "radiusmass3_urgr.pdf")
gui()




## Confrontone

R_nrgr, M_nrgr = Array{Float64}(29), Array{Float64}(29)
for i=1:0.25:8
    h = 0.1
    Umax = 30000.0  # maximum radius considered
    u = linspace(h, Umax, ceil(Umax/h))
    _, __, R_nrgr[Int(i*4-3)], M_nrgr[Int(i*4-3)] = solveLaneEmden(u, 1.5, h, i)
end

R_urgr, M_urgr = Array{Float64}(29), Array{Float64}(29)
for j=1:0.25:8
    h = 0.1
    Umax = 90000.0  # maximum radius considered
    u = linspace(h, Umax, ceil(Umax/h))
    _, __, R_urgr[Int(j*4)-3], M_urgr[Int(j*4)-3] = solveLaneEmden(u, 3.0, h, j)
end

# Physical nonrelativistic radii and mass
R_nr, M_nr = Array{Float64}(29), Array{Float64}(29)
for i=1:0.25:8
    R_nr[Int(i*4)-3], M_nr[Int(i*4)-3] = convertToPhysics(U[1], M[1], 1.5, i)
end

# Physical ultrarelativistic radii and mass
R_ur, M_ur = Array{Float64}(29), Array{Float64}(29)
for j=1:0.25:8
    R_ur[Int(j*4)-3], M_ur[Int(j*4)-3] = convertToPhysics(U[end], M[end], 3.0, j)
end

p5 = plot(R_urgr./1e3, M_urgr, xaxis=("R [km]",(0,80)), yaxis=("Mass [suns]", (0,8)), label=L"n = 3.0\ (TOV)", linewidth=2.5)
plot!(p5, R_ur./1e3, M_ur, label=L"n = 3.0\ (LE)", linewidth=2.5)
plot!(p5, R_nrgr./1e3, M_nrgr, label=L"n = 1.5\ (TOV)", linewidth=2.5)
plot!(p5, R_nr./1e3, M_nr, label=L"n = 1.5\ (LE)", linewidth=2.5)

savefig(p5, "confrontone.pdf")
gui()
