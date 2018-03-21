
using Plots, LaTeXStrings
pyplot(size = (1280, 1080))

function simulation()
    T = 4   # temperature
    N = 256    # number of particles (32, 108, 256...)
    density = 1.3
    nmax = 10000
    dt = 1e-4
    fstep = 20
    E = zeros(Int(nmax/fstep)) # array of total energy
    L = cbrt(N/density)
    X, V = initializeSystem(N, L, T)
    X_ = zeros(3N, round(Int,nmax/fstep)) # storia delle posizioni
    Plots.scatter(X[1:3:3N-2], X[2:3:3N-1], X[3:3:3N], m=(7,0.9,:blue,Plots.stroke(0)),w=7, xaxis=("x",(-L/2,L/2)), yaxis=("y",(-L/2,L/2)), zaxis=("z",(-L/2,L/2)), leg=false)

    @show @elapsed for n = 1:nmax
        F = forces(X,L)
        X, V = velocityVerlet(X, V, F, L, dt)
        if n%fstep == 0
            @show E[Int(n/fstep)] = energy(X,V,L)
            X_[:,Int(n/fstep)] = X
        end
    end

    anim = @animate for i =1:size(X_)[2]-1
        Plots.scatter(X_[1:3:3N-2,i], X_[2:3:3N-1,i], X_[3:3:3N,i], m=(10,0.9,:blue,Plots.stroke(0)),w=7, xaxis=("x",(-L/2,L/2)), yaxis=("y",(-L/2,L/2)), zaxis=("z",(-L/2,L/2)), leg=false)
    end
    filename = string("./LJ_",N,"_T",T,"_d",density,".mp4")
    mp4(anim, filename, fps = 30)
    return anim
end

function makeVideo(pngs)
    filename = string("./LJ_",N,"_T",T,"_d",density,".mp4")
    mp4(anim, filename, fps = 30)
end

function energy(r,v,L)
    T = 0.0
    V = 0.0
    for l=1:Int(length(r)/3)-1
        T += sqrt(v[3l+1]^2 + v[3l+2]^2 + v[3l+3]^2)
        for i=0:l-1
            dx = r[3l+1] - r[3i+1]
            dx = dx - L*round(dx/L)
            dy = r[3l+2] - r[3i+2]
            dy = dy - L*round(dy/L)
            dz = r[3l+3] - r[3i+3]
            dz = dz - L*round(dz/L)
            dr2 = dx*dx + dy*dy + dz*dz
            if dr2 < L*L/4
                V += LJ(sqrt(dr2))
            end
        end
    end
    return T+V
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

function shiftSystem!(A::Array{Float64,1}, L::Float64)
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

LJ(dr::Float64) = -4*(dr^-6 - dr^-12)
der_LJ(dr::Float64) = 4*(6*dr^-8 - 12*dr^-14)

function forces(r::Array{Float64,1}, L::Float64)
    F = zeros(r)
    Threads.@threads for l=1:Int(length(r)/3)-1
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

@fastmath function velocityVerlet(x, v, F, L, dt)
    x_ = Array{Float64}(length(x))
    @. x_ =  x + v*dt + F*dt^2/2
    shiftSystem!(x_, L)
    F_ = forces(x_,L)
    @. v += (F + F_)*dt/2
    return x_, v
end
