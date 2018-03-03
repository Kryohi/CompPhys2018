
using BenchmarkTools, QuadGK, Plots#, PyPlot
#using Gadfly, StatPlots

function integrate(f::Function, x_inf::Float64, x_sup::Float64, N::Int=10^5)  # N è facoltativo
    h = (x_sup-x_inf)/N
    s1 = sum(f.(linspace(x_inf.+2h, x_sup.-2h, (N-2)/2)))   # punti messi per motivi prestazionali
    s2 = sum(f.(linspace(x_inf.+1h, x_sup.-1h, N/2)))
    return (f(x_inf) + f(x_sup) + 2*s1 + 4*s2) * h/3     # Regola di Simpson cubica per il valore dell'integrale
end

function RK3step(f::Function, statevector, h)
    t = statevector[1]
    y = statevector[2]
    k1 =
    k2 =
    k3 =
    y += ()*h/6
    return [t+h, x]
end
