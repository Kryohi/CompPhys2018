
using QuadGK, Plots, Rsvg
#pyplot()
Plots.plotlyjs()

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

h = 1e-4
Umax = 6.0
u = linspace(h, Umax, ceil(Umax/h))
ϕ = zeros(ceil(Umax/h))
θ = zeros(ceil(Umax/h))

## Initial conditions
ϕ[1] = 0.0
θ[1] = 1.0

## solution of the differential system
# u is called ξ on paper
function solveLaneEmden(n::Float64=2.2)  # n is the polytropic index
    f1(u_,phi) = -phi/u_^2
    f2(u_,theta) = u_^2*theta^n
    #for i = 1:length(u)-1
    i = 1
    # Evaluate the differential equations until θ reaches 0
    while θ[i] > 5e-4
        θ[i+1], ϕ[i+1] = RK4system(u[i], f1, f2, θ[i], ϕ[i], h)
        if i > (length(u)-10) println("i = ", i, "  θ = ", θ[i+1], "  ϕ = ", ϕ[i+1]) end
        i+=1
    end
    return [θ, ϕ, u[i]] # where u[i] is the adimensional radius found
end
θ, ϕ, umax = solveLaneEmden(2.1)

## Plots
p1 = plot(u[θ.!=0], θ[θ.!=0])   # plotta ϕ e θ all'interno della stella
plot!(u[θ.!=0], ϕ[θ.!=0])
savefig(p1,"thetaphi.svg")

## Find the radius and mass of the star for a range of n
function radiusMass(nmin, nmax)
    U = []  #raggio adimensionale
    M = []  #massa adimensioanle
    for n = nmin:0.05:nmax
        θ, ϕ, umax = solveLaneEmden(n)
        push!(U, umax)   #  mette in un array il raggio adimensionale raggiunto
        push!(M, 4π*integrateAll(θ.*u.^2, 1e-4))
    end
    return [U,M]
end
U, M = radiusMass(1.5, 3.5)
p2 = plot(U,M)
savefig(p2,"aradiusamass.svg")


## Conversion to physical units
function convertToPhysics(umax,n)
    G = 6.67408e-11
    ρ0 = 0.16e-45
    ρc = 1e-45 # central density, ρ0 < ρc < 8ρ0
    k = 1e20 # ???
    α = k*ρc^(1-1/n)*(n+1)/(4π*G)
    r = α.*u
    rmax = α.*umax
    M = 4π*integrateAll(ρc.*θ.^n.*r.^2, h)
end
