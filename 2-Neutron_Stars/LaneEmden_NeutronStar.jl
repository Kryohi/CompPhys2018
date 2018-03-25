
using Plots, LaTeXStrings
#gr()   #faster than pgfplots()
pyplot()
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
    l1 = fx(t, y)
    k2 = fy(t+h/2, x+l1*h/2)
    l2 = fx(t+h/2, y+k1*h/2)
    k3 = fy(t+h/2, x+l2*h/2)
    l3 = fx(t+h/2, y+k2*h/2)
    k4 = fy(t+h, x+l3*h)
    l4 = fx(t+h, y+k3*h)
    x += (l1 + 2l2 + 2l3 + l4)*h/6
    y += (k1 + 2k2 + 2k3 + k4)*h/6
    return [x, y]
end

## solution of the Lane-Emden differential system
# u is called ξ on paper
function solveLaneEmden(u, n, h=1e-3)  # n is the polytropic index, u the radius array
    umax = u[end]
    ϕ = zeros(ceil(umax/h))
    θ = zeros(ceil(umax/h))

    # Initial conditions
    ϕ[1] = 0.0
    θ[1] = 1.0

    # Differential system
    f1(u_, ϕ_) = -ϕ_ / u_^2  # where ϕ is the derivative of θ times u^2
        f2(u_, θ_) = u_^2 * θ_^n

    # Evaluate the differential equations until θ reaches 0 (almost)
    i = 1
    while θ[i]^n > 1e-6
        θ[i+1], ϕ[i+1] = RK4system(u[i], f1, f2, θ[i], ϕ[i], h)
        i+=1
    end
    return [θ, ϕ, u[i]] # where u[i] is the adimensional radius found
end

## plot θ e ρ inside the star for different ns
function densityProfiles(nn)
    h = 1e-4
    Umax = 10.0  # maximum adimensional radius considered
    u = linspace(h, Umax, ceil(Umax/h))
    θ, ϕ, = solveLaneEmden(u, nn[1], h)
    p1 = plot(u[1:5:end], θ[1:5:end], xlab=L"\xi", ylab=L"\theta", label=latexstring("n = ",nn[1]), xaxis=((0.0, 6.01)))
    p2 = plot(u[1:5:end], θ[1:5:end].^nn[1], xlab=L"\xi", ylab=L"\rho/\rho_c", label=latexstring("n = ",nn[1]))
    p3 = plot(u[1:5:end], ϕ[1:5:end].*u[1:5:end].^(-2), xlab=L"\xi", ylab=L"\frac{d\theta}{d\xi}", label=latexstring("n = ",nn[1]))
    for n ∈ nn[2:end]
        θ, ϕ, = solveLaneEmden(u, n, h)
        plot!(p1, u[1:5:end], θ[1:5:end], label=latexstring("n = ",n))
        plot!(p2, u[1:5:end], θ[1:5:end].^n, label=latexstring("n = ",n))
        plot!(p3, u[1:5:end], ϕ[1:5:end].*u[1:5:end].^(-2), label=latexstring("n = ",n))
    end
    Plots.savefig(p1,"theta.pdf")    #better to use with the pgfplots() backend
    Plots.savefig(p2,"density.pdf")
    Plots.savefig(p3,"dtheta.pdf")
end

densityProfiles(1.5:0.3:3.0)

## Find the radius and mass of the star for a range of n (not yet in physical units)
function radiusMass(nn)
    U = []  # raggio adimensionale
    M = []  # massa adimensioanle
    h = 1e-4
    Umax = 10.0  # maximum adimensional radius
    u = linspace(h, Umax, ceil(Umax/h))
    for n ∈ nn
        θ, ϕ, R = solveLaneEmden(u, n, h)
        push!(U, R)   #  mette in un array il raggio adimensionale raggiunto
        push!(M, 4π*integrateAll(θ.^n.*u.^2, h))
    end
    return [U,M]
end

U, M = radiusMass(1.5:0.1:3.0)

p4 = scatter(1.5:0.1:3.0, U, leg=false, m=(5,0.7,:blue,Plots.stroke(0)),w=5,xaxis=("n",(1.45,3.05)), yaxis=("Adimensional radius",(3.5,6.0)))
Plots.savefig(p4,"raggi.pdf")
p5 = Plots.plot(U, M, xaxis=("Adimensional radius",(3.65,5.5)), yaxis=("Adimensional mass",(26,36)), leg=false)
savefig(p5,"aradiusamass.pdf")


## Conversion to physical units
function convertToPhysics(umax, mass, n, fattoremisterioso)
    G = 6.67408e-11
    mn = 1.674927351e-27 # neutron mass
    ρ0 = 0.16e45*mn
    ρc = ρ0*fattoremisterioso # central density, ρ0 < ρc < 8ρ0
    ħ = 1.0545718e-34
    c = 299792458
    if n == 3.0
        K = c*ħ/4*(3*π^2)^(1/3)   # ultrarelativistic
    elseif n == 1.5
        K = (3*π^2)^(2/3)*ħ^2/(5*mn)      # non-relativistic #was ^(1/3)
    end
    #K = 10.2*(1.9885e30)^(-2/3)*1e6
    #@show K = h^2/(15*π^2*mn)(3π^2/(mn*c^2))^(5/3)
    α = sqrt(K*(ρc/mn)^(1+1/n)*(n+1)/(4π*G*ρc^2))
    Rmax = α*umax
    Ms = 1.9885e30 # solar mass (kg)
    Mass = α^3*ρc*mass/Ms
    return [Rmax, Mass]
end


## Plots of the radii and mass varying ρc (fattoremisterioso), in the relativistic and non-relativistic case

# Physical nonrelativistic radii and mass
R_nr, M_nr = Array{Float64}(8), Array{Float64}(8)
for i=1:8
    R_nr[i], M_nr[i] = convertToPhysics(U[1], M[1], 1.5, i)
end
p6 = Plots.scatter(R_nr./1e3, M_nr, xaxis=("R [km]",(12.5, 20)), yaxis=("Mass [suns]",(0.5, 1.7)), leg=false,m=(7,0.7,:blue,Plots.stroke(0)))
savefig(p6,"radiusmass2_nr.pdf")

# Physical ultrarelativistic radii and mass
R_ur, M_ur = Array{Float64}(8), Array{Float64}(8)
for j=1:8
    R_ur[j], M_ur[j] = convertToPhysics(U[end], M[end], 3.0, j)
end

p8 = Plots.scatter(R_ur./1e3, M_ur, xaxis=("R [km]",(30, 72)), yaxis=("Mass [suns]",(5.0, 6.0)), leg=false,m=(7,0.7,:blue,Plots.stroke(0)))
savefig(p8, "radiusmass2_ur.pdf")
