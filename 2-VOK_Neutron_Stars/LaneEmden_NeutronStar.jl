
using QuadGK, Plots #, PyPlot
#using Gadfly, StatPlots

function RK4step(f::Function, t, y, h=1e-6)
    @show k1 = f(t, y)
    @show k2 = f(t+h/2, y+k1*h/2)
    @show k3 = f(t+h/2, y+k2*h/2)
    @show k4 = f(t+h, y+k3*h)
    @show y += (k1 + 2k2 + 2k3 + k4)*h/6
    return y
end

n = 1.5 # polytropic index
h = 1e-4
Ymax = 10
u = linspace(0, Ymax, ceil(Ymax/h))
ϕ = zeros(ceil(Ymax/h))
θ = zeros(ceil(Ymax/h))

## Initial conditions
ϕ[1] = 0.0
θ[1] = 1.0
f1(y,f) = -f/y^2
f2(y,f) = y^2*f^n
for i = 1:2#length(u)-1
    θ[i+1] = RK4step(f1, u[i], ϕ[i], h)
    ϕ[i+1] = RK4step(f2, u[i], θ[i], h)
end

pyplot()
plot(u,θ)
gui()

## Conversion to physical units
G = 6.67408e-11
ρ0 = 0.16e-45
ρc = 0.2e-45 # central density, ρ0 < ρc < 8ρ0
k = 2 # ???
α = k*ρc^(1-1/n)*(n+1)/(4π*G)
r = α.*u


## Test vari
# t = 0:1e-5:1
# V = zeros(1000)
# Y = zeros(1000)
# V[1] = 4.0
# Y[1] = 0.0
# gravity(t, y) = -9.8
# for i=1:length(V)-1
#     V[i+1] = RK4step(gravity, t[i], V[i], 1e-5)
#     newton(t,y) = V[i]
#     Y[i+1] = RK4step(newton, t[i], Y[i], 1e-5)
# end
#
# plot(t,Y)
# gui()
