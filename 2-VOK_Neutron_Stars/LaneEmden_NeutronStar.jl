
using QuadGK, Plots #, PyPlot
#using Gadfly, StatPlots

function RK3step(f::Function, statevector, h)
    t = statevector[1]
    y = statevector[2]
    k1 =
    k2 =
    k3 =
    y += ()*h/6
    return [t+h, x]
end



## Conversion to physical units
G = 6.67408*10^-11
ρ0 = 0.16e-45
ρc = 0.2e-45 # central density, ρ0 < ρc < 8ρ0
α = k*ρc^(1-1/n)*(n+1)/(4π*G)
r = αy
