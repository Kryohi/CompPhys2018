
using Plots, LaTeXStrings, ProgressMeter
pyplot(size = (1280, 1080))
fnt = "Source Sans Pro"
default(titlefont=Plots.font(fnt, 16), guidefont=Plots.font(fnt, 16), tickfont=Plots.font(fnt, 12), legendfont=Plots.font(fnt, 12))

# Main function, it creates the initial systems, runs it through the Verlet algorithm for maxsteps
# and optionally creates an animation of the particles (also doable at a later time from the XX output)
function simulation(; N::Int=256, T::Float64=4, rho::Float64=1.3, maxsteps = 5000, animation=false)
    dt = 1e-4
    fstep = 20
    E = zeros(Int(maxsteps/fstep)+1) # array of total energy
    L = cbrt(N/rho)
    X, V = initializeSystem(N, L, T)
    XX = zeros(3N, Int(maxsteps/fstep)+1) # storia delle posizioni

    @time for n = 1:maxsteps
        F = forces(X,L)
        X, V = velocityVerlet(X, V, F, L, dt)
        if n%fstep == 0
            E[Int(n/fstep)+1] = energy(X,V,L)
            XX[:,Int(n/fstep)+1] = X
        end
    end
    if animation
        makeVideo(XX, N=N, T=T, rho=rho)
    end

    return XX, E   # returns a matrix with the hystory of positions, plus the energy and pressure arrays
end


# Initialize the system at t=0 as a perfect FCC crystal centered in 0, plus adimensional
# maxwell-boltzmann velocities
function initializeSystem(N::Int, L, T)
    Na = round(Int,∛(N/4)) # number of cells per dimension
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

function forces(r::Array{Float64}, L::Float64)
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

function velocityVerlet(x, v, F, L, dt)
    x_ = Array{Float64}(length(x))
    @. x_ =  x + v*dt + F*dt^2/2
    shiftSystem!(x_, L)
    F_ = forces(x_,L)
    @. v += (F + F_)*dt/2
    return x_, v
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

# creates an array with length N of gaussian distributed numbers, with σ = sigma
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

# fare attenzione a non prendere particella vicino ai bordi
function lindemann(X0, XX, N, rho)   # where X0 is a triplet at t=0, XX the hystory of that point at t>0
    L = cbrt(N/rho)
    Na = round(Int,∛(N/4)) # number of cells per dimension
    a = L / Na  # passo reticolare
    deltaX = sqrt(sum((XX[1,:]-X0[1]).^2 .+ (XX[2,:]-X0[2]).^2 .+ (XX[3,:]-X0[3]).^2) / (length(XX[1,:])-1))
    return deltaX*2/a
end


# makes an mp4 video made by a lot of 3D plots (can be easily modified to produce a gif instead)
function makeVideo(X_; N="???", T="???", rho="???")
    L = cbrt(N/rho)
    prog = Progress(size(X_)[2],1)  # initialize progress bar
    println("\nI'm cooking pngs to make a nice video. It will take some time...")

    anim = @animate for i =1:size(X_)[2]
        Plots.scatter(X_[1:3:3N-2,i], X_[2:3:3N-1,i], X_[3:3:3N,i], m=(10,0.9,:blue,Plots.stroke(0)),w=7, xaxis=("x",(-L/2,L/2)), yaxis=("y",(-L/2,L/2)), zaxis=("z",(-L/2,L/2)), leg=false)
        next!(prog) # increment the progress bar
    end
    filename = string("/home/kryohi/Video/LJ_",N,"_T",T,"_d",rho,".mp4")
    mp4(anim, filename, fps = 30)
    gui()
end

function make3Dplot(A::Array{Float64}; L=-1.0)
    if L == -1.0
        Plots.scatter(A[1:3:3N-2], A[2:3:3N-1], A[3:3:3N], m=(7,0.9,:blue,Plots.stroke(0)),w=7, xaxis=("x"), yaxis=("y"), zaxis=("z"), leg=false)
    else
        Plots.scatter(A[1:3:3N-2], A[2:3:3N-1], A[3:3:3N], m=(7,0.9,:blue,Plots.stroke(0)),w=7, xaxis=("x",(-L/2,L/2)), yaxis=("y",(-L/2,L/2)), zaxis=("z",(-L/2,L/2)), leg=false)
    end
    gui()
end
