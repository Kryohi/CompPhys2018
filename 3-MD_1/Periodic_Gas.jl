srand(420) # setting the seed, to make the random numbers reproducible

function boxMuller(sigma, x0=0)
    @show x1, x2 = rand(2)
    @show c1 = sqrt(2sigma*log(1-x1)*cos(2π*x2))
    #c2 = sqrt(2sigma*log(1-x2)*cos(2π*x1))
    return exp(-(c1 - x0)/2sigma^2)/sqrt(2π*sigma^2)
end

function initializeSystem(N::Int, density, T)
    L = (N/density)^(1/3)
    σ = T*4.2
    X = rand(3N).*L - L/2
    V = Array{Float64}(3N)
    V .= boxMuller(σ)
    return [X, V]
end

XX, VV = initializeSystem(10, 100, 1)
