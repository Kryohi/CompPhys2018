
using Plots, LaTeXStrings, BenchmarkTools
pyplot()

function simulation()
    T = 5   # temperature
    N = 32    # number of particles
    density = 0.1
    nmax = 2000
    global L = cbrt(N/density)
    X, V = initializeSystem(N, L, T)
    X_ = zeros(3N, round(Int,nmax/20)) # storia delle posizioni

    for n = 1:nmax
        F = forces(X)
        X, V = velocityVerlet(X, V, F, 1e-4)
        if n%20 == 0
            X_[:,round(Int,n/20)] = X
        end
    end
    size(X_)[2]
    @gif for i =1:size(X_)[2]
        scatter(X_[1:3:3N-2,i], X[2:3:3N-1,i], X[3:3:3N,i], m=(7,0.9,:blue,Plots.stroke(0)),w=7, xaxis=("x",(-L/2,L/2)), yaxis=("y",(-L/2,L/2)), zaxis=("z",(-L/2,L/2)), leg=false)
    end
    gui()
end

function boxMuller(sigma, N::Int, x0=0.0)
    srand(42)   # sets the rng seed, to obtain reproducible numbers
    c = Array{Float64}(N)
    for j = 1:round(Int,N/2)
        x1, x2 = rand(2)
        c[j*2] = sqrt(-2sigma*log(1-x1))*cos(2π*x2)
        c[j*2-1] = sqrt(-2sigma*log(1-x2))*sin(2π*x1)
    end
    return c
end
function vecboxMuller(sigma, N::Int, x0=0.0) #should be ~50% faster
    srand(42)   # sets the rng seed, to obtain reproducible numbers
    x1 = rand(Int(N/2))
    x2 = rand(Int(N/2))
    @. [sqrt(-2sigma*log(1-x1))*cos(2π*x2); sqrt(-2sigma*log(1-x2))*sin(2π*x1)]
end

function shiftSystem!(A,L)
    for j in eachindex(A)
        @inbounds A[j] = A[j] - L*round(A[j]/L)
    end
end

function initializeSystem(N::Int, L, T)
    Na = round(Int,∛(N/4)) # numero celle per dimensione
    a = L / Na  # passo reticolare
    if Na - ∛(N/4) != 0
        error("Can't make a cubic FCC crystal with this number of particles :(")
    end

    X = Array{Float64}(3N)
    for i=0:Na-1, j=0:Na-1, k = 0:Na-1  # loop over every cell of the cfc lattice
        n = i*Na*Na + j*Na + k # unique number for each triplet i,j,k
        X[n*12+1], X[n*12+2], X[n*12+3] = a*i, a*j, a*k # vertice celle [x1,y1,z1,...]
        X[n*12+4], X[n*12+5], X[n*12+6] = a*i + a/2, a*j + a/2, a*k
        X[n*12+7], X[n*12+8], X[n*12+9] = a*i + a/2, a*j, a*k + a/2
        X[n*12+10], X[n*12+11], X[n*12+12] = a*i, a*j + a/2, a*k + a/2
    end
    X += a/4   # needed to avoid particles exactly at the edges of the box
    shiftSystem!(X,L)
    σ = T     # T * k_b /m in qualche unità di misura
    V = boxMuller(σ,3N)  # da sistemare per array
    return [X, V]
end

der_LJ(dr) = 4*(6*dr^-8 - 12*dr^-14)

function forces(r)
    F = zeros(length(r))
    for l=1:Int(length(r)/3)-1
        for i=0:l-1
            dx = r[3l+1] - r[3i+1]
            dx = dx - L*round(dx/L)
            dy = r[3l+2] - r[3i+2]
            dy = dy - L*round(dy/L)
            dz = r[3l+3] - r[3i+3]
            dz = dz - L*round(dz/L)
            dr2 = dx*dx + dy*dy + dz*dz
            if dr2 < L*L/4
                dV = -der_LJ(sqrt(dr2))
                F[3l+1] += dV*dx
                F[3l+2] += dV*dy
                F[3l+3] += dV*dz
                F[3i+1] -= dV*dx
                F[3i+2] -= dV*dy
                F[3i+3] -= dV*dz
            end
        end
    end
    return F
end

function velocityVerlet(x, v, F, dt)
    x_ = Array{Float64}(length(x))
    @. x_ =  x + v*dt + F*dt^2/2
    shiftSystem!(x_, L)
    F_ = forces(x_)
    @. v += (F + F_)*dt/2
    return x_, v
end
