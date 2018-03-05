
using QuadGK, Plots #, PyPlot
pyplot()
#using Gadfly, StatPlots

function RK4single(t, f, y, h=1e-3)
    @show k1 = f(t, y)
    k2 = f(t+h/2, y+k1*h/2)
    k3 = f(t+h/2, y+k2*h/2)
    k4 = f(t+h, y+k3*h)
    @show y += (k1 + 2k2 + 2k3 + k4)*h/6
    return y
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

n = 2.5 # polytropic index
h = 1e-4
Ymax = 6.0
u = linspace(h, Ymax, ceil(Ymax/h))
ϕ = zeros(ceil(Ymax/h))
θ = zeros(ceil(Ymax/h))

## Initial conditions
ϕ[1] = 0.0
θ[1] = 1.0

## solution of the differential system
# u and y are the same variable, called ξ on paper
f1(y,phi) = -phi/y^2
f2(y,theta) = y^2*theta^n
#for i = 1:length(u)-1
i = 1
while θ[i] > 5e-4
    θ[i+1], ϕ[i+1] = RK4system(u[i], f1, f2, θ[i], ϕ[i], h)
    if i > (length(u)-10) println("i = ", i, "  θ = ", θ[i+1], "  ϕ = ", ϕ[i+1]) end
    i+=1
    #θ[i+1] = RK4step(f1, u[i], ϕ[i], h)
end
umax = u[i]

## Plots
plot(u,θ)
plot!(u,ϕ)
gui()

## Conversion to physical units
G = 6.67408e-11
ρ0 = 0.16e-45
ρc = 0.2e-45 # central density, ρ0 < ρc < 8ρ0
k = 2 # ???
α = k*ρc^(1-1/n)*(n+1)/(4π*G)
r = α.*umax


## Test vari
# function RK4osc(f, t, v, x, h=1e-3)
#     k1 = f(t, x)
#     k2 = f(t+h/2, x+k1*h/2)
#     k3 = f(t+h/2, x+k2*h/2)
#     k4 = f(t+h, x+k3*h)
#     v += (k1 + 2k2 + 2k3 + k4)*h/6
#     x += v*h
#     return [v, x]
# end
# h = 1e-3
# Tmax = 5
# t = linspace(h, Tmax, ceil(Tmax/h))
# V = zeros(ceil(Tmax/h))
# Y = zeros(ceil(Tmax/h))
# V[1] = 1.0
# Y[1] = 3.0
# molla(t, y) = -2*y
# pos(t, v) = v
# for i=1:(length(t)-1)
#     V[i+1], Y[i+1] = RK4osc(molla, t[i], V[i], Y[i], h)
#     println("vx = ", V[i+1], "  x = ", Y[i+1])
# end
#
# plot(t,Y)
# plot!(t,V)
# gui()
