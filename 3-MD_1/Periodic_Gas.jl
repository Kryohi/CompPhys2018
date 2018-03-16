
using Plots, LaTeXStrings
pyplot()
srand(420) # setting the seed, to make the random numbers reproducible

function boxMuller(sigma, x0=0)
    @show x1, x2 = rand(2)
    @show c1 = sqrt(2sigma*log(1-x1)*cos(2π*x2))
    #c2 = sqrt(2sigma*log(1-x2)*cos(2π*x1))
    return exp(-(c1 - x0)/2sigma^2)/sqrt(2π*sigma^2)
end

# function shiftSystem!(A)
#     A - L*round(A/L)
# end

function initializeSystem(N::Int, density, T)
    L = (N/density)^(1/3)
    σ = T*4.2 #da sistemare
    #X = rand(3N).*L - L/2 # posizioni random dentro la scatola
    X = Array{Float64}(3N)
    Na = round(Int,(N/4)^(1/3)) # numero celle per dimensione
    a = L / Na  # passo reticolare
    for i=0:Na-1, j=0:Na-1, k = 0:Na-1  # loop over every cell of the cfc lattice
        n = i*Na*Na + j*Na + k # unique number for each triplet i,j,k
        X[n*4+1], X[N+n*4+1], X[2N+n*4+1] = a*i, a*j, a*k # vertice celle
        X[n*4+2], X[N+n*4+2], X[2N+n*4+2] = a*i + a/sqrt(2), a*j + a/sqrt(2), a*k
        X[n*4+3], X[N+n*4+3], X[2N+n*4+3] = a*i + a/sqrt(2), a*j, a*k + a/sqrt(2)
        X[n*4+4], X[N+n*4+4], X[2N+n*4+4] = a*i, a*j + a/sqrt(2), a*k + a/sqrt(2)
    end
    #shiftSystem!(X)
    V = Array{Float64}(3N)
    V .= boxMuller(σ)
    return [X, V]
end

XX, VV = initializeSystem(32, 32, 1)
scatter(XX[1:32], XX[33:64], XX[65:96],m=(7,0.9,:blue,Plots.stroke(0)),w=7, xaxis=("x",(0,1)), yaxis=("y",(0,1)), zaxis=("z",(0,1)))
gui()
