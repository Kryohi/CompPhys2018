
using Plots, LaTeXStrings
#gr()   #faster than pgfplots()
pyplot() #xaxis=(0.0,Umax), yaxis=(0.0,1.05) plot!(u[θ.!=0], ϕ[θ.!=0])
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
    f1(u_,phi) = -phi/u_^2
    f2(u_,theta) = u_^2*theta^n

    # Evaluate the differential equations until θ reaches 0 (almost)
    i = 1
    while θ[i]^n > 1e-4
        θ[i+1], ϕ[i+1] = RK4system(u[i], f1, f2, θ[i], ϕ[i], h)
        i+=1
    end
    return [θ, ϕ, u[i]] # where u[i] is the adimensional radius found
end

## plot θ e ρ inside the star for different ns
function densityProfiles(nn)
    h = 5e-4
    Umax = 7.0  # maximum adimensional radius considered
    u = linspace(h, Umax, ceil(Umax/h))
    θ, ϕ, = solveLaneEmden(u, nn[1], h)
    p1 = plot(u[1:5:end], θ[1:5:end], xlab=L"\xi", ylab=L"\theta", label=latexstring("n = ",nn[1]), xaxis=((0.0, 6.01)))
    p2 = plot(u[1:5:end], θ[1:5:end].^nn[1], xlab=L"\xi", ylab=L"\rho", label=latexstring("n = ",nn[1]))
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
    h = 5e-4
    Umax = 6.0  # maximum adimensional radius
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
    ρ0 = 0.16e45*1.674927351e-27
    ρc = ρ0*fattoremisterioso # central density, ρ0 < ρc < 8ρ0
    K = 10.2*(1.9885e30)^(-2/3)*1e6
    @show α = sqrt(K*ρc^(1-1/n)*(n+1)/(4π*G))
    #r = α.*u
    Rmax = α.*umax
    @show Mass = α.^3.*ρc.*mass
    return [Rmax, Mass]
end

Rmax, Mass = convertToPhysics(U[2], M[2], 1.6, 1)


## TOV
