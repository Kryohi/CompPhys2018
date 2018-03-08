## TODO
#? ciclo sui γ
# precisione numeri maggiore per livelli n alti (provare DoubleDouble?)
# cercare modo per rendere i cicli interni o esterni multithread
#! stoppare ricerca n se S a estremi non ha più segno opposto

using QuadGK, Plots, ColorSchemes, LaTeXStrings

gamma = [21.7, 24.8, 150]   # H2-H2, H2-O2, O2-O2

## Metodi vari
# function that creates a range with exponentially spaced items
function log2space(x1, x2, N)  # todo: preallocate memory with where
    if x1>0 && x2>0
        return 2.^linspace(log2(x1), log2(x2), N)
    elseif x1<0 && x2<0
        return -2.^linspace(log2(-x1), log2(-x2), N)
    else
        warn("Boundary signs don't match, or maybe the culprit is a rogue zero")
        return 0
    end
end

# radice magica per imbrogliare gli estremi imperfetti di integrazione
function sqrtmod(x::Float64)
    if x<=0 return 0.0
    else return sqrt(x)
    end
end

function integrate(f::Function, x_inf::Float64, x_sup::Float64, N::Int=5*10^4)  # N è facoltativo
    h = (x_sup-x_inf)/N
    s1 = sum(f.(linspace(x_inf.+2h, x_sup.-2h, (N-2)/2)))   # punti messi per motivi prestazionali
    s2 = sum(f.(linspace(x_inf.+1h, x_sup.-1h, N/2)))
    return (f(x_inf) + f(x_sup) + 2*s1 + 4*s2) * h/3     # Regola di Simpson cubica
end

function zerosecant(f::Function, x1::Float64, x2::Float64, inf::Float64=-1e-10, sup::Float64=1e-10)
    if f(x1)==0
        return x1
    elseif f(x2)==0
        return x2
    elseif f(x1)*f(x2)>0
        println("f(x1) = ", f(x1), "  f(x2) = ", f(x2))
        error("f(X1) and f(X2) must have an opposing sign")
        return -1
    else
        while f(x2)<inf || f(x2)>sup  # restituisce il valore dove la funzione fa circa 0
            x_ = x2  # x precedente
            x2 = x2 - f(x2)*(x2-x1)/(f(x2)-f(x1))
            x1 = x_
        end
    end
    return x2
end

function zerosecant(f::Function, x1::Array{Float64}, x2::Array{Float64}, inf::Float64=-1e-10, sup::Float64=1e-10)
    for i = 1:length(x1)
        if f(x1[i])*f(x2[i])>0
            println("f(x1) = ", f(x1[i]), "  f(x2) = ", f(x2[i]))
            error("f(X1) and f(X2) must have an opposing sign")
            return -1
        end
        while f(x2[i])<inf || f(x2[i])>sup  # metodo della secante per trovare lo zero
            x_ = x2[i]  # x precedente
            x2[i] = x2[i] - f(x2[i])*(x2[i]-x1[i])/(f(x2[i])-f(x1[i]))
            x1[i] = x_
        end
    end
    return x2
end

# trova tutti (in teoria 2) gli intervalli [x1,x2] dove V-E cambia di segno
function xZeros(f::Function, X1::Float64, X2::Float64, N::Int=256; mode="normal")    # N e mode argomenti opzionali
    x0, x1, x2, x = Array{Float64}, Array{Float64}, Array{Float64}, Array{Float64}(N)
    if mode=="normal"
        h = (X2-X1)/N
        x = linspace(X1, X2, N) # N-1?
        @inbounds I = find(sign.(f.(x[1:end-1])) .- sign.(f.(x[2:end])) .≠ 0)
        x1, x2 = x[I], x[I.+1]
        # x1 = x[sign.(f.(x)) .- sign.(f.(x.+h)) .≠ 0]; x2 = x1 .+ h
    else
        x = log2space(X1, X2, N)
        @inbounds I = find(sign.(f.(x[1:end-1])) .- sign.(f.(x[2:end])) .≠ 0)
        x1, x2 = x[I], x[I.+1]
    end
    if isempty(x1)
        error("Oh shit, can't find the zeros needed for the integration...")
    end
    if length(x1)==1
        warn("I'm seeing only one zero where there should be two, maybe I should drink something to fix this")
        @show x1 = [x1[1], zerosecant(f, x1[1], x2[1], -1e-13, 1e-10)+0.001]
        @show x2 = [x2[1], 4200.0]
    end
    x0 = zerosecant(f, x1, x2, -1e-13, 1e-10)
    return sort(x0)
end

# trova tutti (in teoria 1) gli intervalli [x1,x2] dove S(E) cambia di segno
function EZero(f::Function, X1::Float64, X2::Float64, N::Int=16; mode="normal")
    #x0, x1, x2 = Float64, Float64, Float64 #,1
    if mode == "normal"
        h = (X2-X1)/N
        x = (X2):-h:(X1+h)
        x1 = x[sign.(f.(x)) .- sign.(f.(x.-h)) .≠ 0]
        x2 = x1 - h
    else
        cut = X1/4
        x = vcat(linspace(X1, cut, ceil(N/2)), log2space(cut, X2, N))
        I = find(sign.(f.(x[1:end-1])) .- sign.(f.(x[2:end])) .≠ 0)
        x1, x2 = x[I], x[I.+1]
    end
    if isempty(x1)
        warn("No zeros of S were found, trying with the passed range...")
        @show x1, x2 = X1, X2
        @show f(x1), f(x2)
    end
    if length(x1)>1
        warn("I don't know what's going on here... but trying to continue")
    end
    #@show x0 = zerosecant(f, x1[1], x2[1])
    return zerosecant(f, x1[1], x2[1])
end

## Vogliamo trovare le E per cui action si azzera
# S dipende da un integrale i cui estremi xin e xout dipendono a loro volta da E
function S(E::Float64, n::Int, γ, xMin=0.97, xMax=16.0)

    intgr(x) = sqrtmod(E - 4*(x^-12-x^-6))   # argomento dell'integrale di azione da usare dopo
    X = xZeros.(x -> E - 4*(x^-12-x^-6), xMin, xMax, 200, mode="log")    # trova xin e xout
    if isnan(E) error("E non è più un numero, viva la libertà!") end

    if length(X) >= 2
        #println("E = ", E, "; A^2 at boundaries: ", E - 4*(X[1]^-12 -X[1]^-6), "   ", E - 4*(X[2]^-12 - X[2]^-6))
        #println("\nBetween ", X[1]+1e-10, " and ", X[2]-1e-10, ":\n  S = ", γ*integrate(intgr, X[1]+1e-10, X[2]-1e-10) - n*π)
        #return γ*integrate(intgr, X[1], X[end]) - n*π  # dominio ristretto di 1e-9 perché se no boh
        return γ*quadgk(intgr, X[1], X[end], order=8, maxevals=10^8)[1] - n*π   #così si imbroglia però
    else
        warn("Less than 2 zeros found, xout and/or xin are missing (", X, "  ", E, ")")
        return golden/42   # must be positive, because until veeeery small E, S(E) is still negative
    end
end

##  Troviamo tutti (se ciao) i livelli energetici discreti
# dopo il punto e virgola keyword arguments, solo per chiarezza
function findSpectrum(max::Int, γ; Emin=-0.95, Emax=-1e-5, xMin=0.95, xMax=16.0)
    energy = zeros(max)
    setmode="normal"
    slices = 12
    for n = 1:max
        if n>4 setmode="log" end
        energy[n] = EZero(E -> S(E, n, γ, xMin, xMax), Emin, Emax, slices, mode=setmode)
    end
    return energy
end

@time @show E = findSpectrum(5, gamma[1], Emin=-0.95, Emax=-1e-7, xMin=0.97, xMax=24.0)
# con Float64 al massimo si arriva a 5 livelli con γ1, 6 con γ2 e circa 40 con γ3
# a E=-0.00001 xout è circa 8.6, a -0.000002 circa 11.2, -10^-8 circa 27

## Grafici
pgfplots()
function plotLevels(f, E)
    X = linspace(0.94, 3.0, 8000)
    V = f.(X)
    Plots.plot(X, V, label="V", xaxis=("x",(0.9,3.0)), yaxis=("E",(-1.08,2.55)), linewidth=2, linecolor=RGB(0.3,0.8,0.4))
    C = RGB[ColorSchemes.inferno[floor(Int,z)] for z=linspace(100,180,length(E))]
    for n=1:length(E)
        #labl = string("E_", n)
        Plots.hline!([E[n]], label=latexstring("E_",n), linecolor=C[n],linewidth=0.5)
    end
    savefig("energylevels1.pdf")
    savefig("energylevels1.svg")
end
V(x) = 4*(x^-12-x^-6)
plotLevels(V, E)
