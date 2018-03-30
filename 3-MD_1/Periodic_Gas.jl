## TODO
# Capire discrepanza T calcolata e inizializzata
# controllare mezzi in pressione, energia, temperatura
# velocityverlet! con where{}
# finire Prettyprint
# parametro d'ordine
# risolvere CM vagabondo (in realtà è abbastanza fermo)
# velocizzare creazione animazione o minacciare maintainer su github di farlo
# aggiungere entropia, energia libera di Gibbs...
# usare Measurements per variabili termodinamiche medie
# provare Gadfly

using Plots, ProgressMeter, DataFrames, CSV
pyplot(size = (1280, 1080))
fnt = "Source Sans Pro"
default(titlefont=Plots.font(fnt, 18), guidefont=Plots.font(fnt, 18), tickfont=Plots.font(fnt, 14), legendfont=Plots.font(fnt, 14))

# Main function, it creates the initial system, runs it through the Verlet algorithm for maxsteps,
# saves the positions arrays every fstep iterations, returns and saves it as a csv file
# and optionally creates an animation of the particles (also doable at a later time from the XX output)
function simulation(; N=256, T0=4.0, rho=1.3, dt = 1e-4, fstep = 50, maxsteps = 10^4, anim=false, csv=true, simonly=false)

    L = cbrt(N/rho)
    X, V = initializeSystem(N, L, T0)
    XX = zeros(3N, Int(maxsteps/fstep)) # storia delle posizioni
    E = zeros(Int(maxsteps/fstep)) # array of total energy
    T = zeros(E) # array of temperature
    P = zeros(E) # array of total pressure
    CM = zeros(3*Int(maxsteps/fstep)) # da togliere
    F = forces(X,L) # initial forces
    #make3Dplot(V,L)
    println()

    prog = Progress(maxsteps, dt=1.0, desc="Simulating...", barglyphs=BarGlyphs("[=> ]"), barlen=50)
    for n = 1:maxsteps
        if !simonly && n%fstep == 1
            i = ceil(Int,n/fstep)
            E[i] = energy(X,V,L)
            T[i] = temperature(V)
            P[i] = vpressure2(X,F,L) + T[i]*rho
            #CM[3i-2:3i] = avg3D(X)
            XX[:,i] = X
        end
        X, V, F = velocityVerlet(X, V, F, L, dt)
        next!(prog)
    end
    csv && saveCSV(XX', N=N, T=T0, rho=rho)
    anim && makeVideo(XX, T=T0, rho=rho)

    prettyPrint(L, rho, X, V, E, T, P, CM)
    return XX, E, T, P, CM # returns a matrix with the hystory of positions, energy and pressure arrays
end

# Initialize the system at t=0 as a perfect FCC crystal centered in 0, plus adimensional
# maxwell-boltzmann velocities
function initializeSystem(N::Int, L, T)
    Na = round(Int,cbrt(N/4)) # number of cells per dimension
    a = L / Na  # passo reticolare
    !isapprox(Na, cbrt(N/4)) && error("Can't make a cubic FCC crystal with this N :(")

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
    σ = sqrt(T)     #  in qualche unità di misura
    V = vecboxMuller(σ,3N)
    #@show temperature(V)
    # force the average velocity to 0
    V[1:3:N-2] -= 3*sum(V[1:3:N-2])/N   # capire perch serve il 3 e perch T cambia
    V[2:3:N-1] -= 3*sum(V[2:3:N-1])/N
    V[3:3:N] -= 3*sum(V[3:3:N])/N
    #@show [sum(V[1:3:N-2])/N, sum(V[2:3:N-1])/N, sum(V[3:3:N])/N]
    #@show temperature(V)
    return [X, V]
end


@inbounds function velocityVerlet(x, v, F, L, dt)
    x_ = Array{Float64}(length(x))
    @. x_ =  x + v*dt + F*dt^2/2
    shiftSystem!(x_, L)
    F_ = forces(x_,L)
    @. v += (F + F_)*dt/2
    return x_, v, F_
end
function velocityVerlet!(x::Array{T}, v::Array{T}, F::Array{T}, L::Float64, dt::Float64) where T #TEST
    @. x += v*dt + F*dt^2/2
    shiftSystem!(x, L)
    F_ = forces(x,L)
    @. v += (F + F_)*dt/2
    return x, v, F_
end # non ancora usato


LJ(dr::Float64) = 4*(dr^-12 - dr^-6)
der_LJ(dr::Float64) = 4*(6*dr^-8 - 12*dr^-14)

@inbounds function forces(r::Array{Float64,1}, L::Float64)
    F = zeros(r)
    # multithreading is convenient only for large N
    # to set the number of threads use the environment variable JULIA_NUM_THREADS=4
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


function avg3D(A::Array{Float64,1})
    N = Int(length(A)/3)
    return [sum(A[1:3:N-2])/N, sum(A[2:3:N-1])/N, sum(A[3:3:N])/N]
end

function energy(r,v,L)
    T = (v[1]^2 + v[2]^2 + v[3]^2)/2 #perché nel ciclo sotto il primo elemento non verrebbe considerato
    V = 0.0
    @inbounds for l=1:Int(length(r)/3)-1
        T += (v[3l+1]^2 + v[3l+2]^2 + v[3l+3]^2)./2
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

function vpressure(r,L) # non usato e probabilmente sbagliato
    P = 0.0
    for l=1:Int(length(r)/3)-1
        @inbounds for i=0:l-1
            dx = r[3l+1] - r[3i+1]
            dx = dx - L*round(dx/L)
            dy = r[3l+2] - r[3i+2]
            dy = dy - L*round(dy/L)
            dz = r[3l+3] - r[3i+3]
            dz = dz - L*round(dz/L)
            dr = sqrt(dx^2 + dy^2 + dz^2)
            if dr < L/2
                P += der_LJ(dr)*dr^2
            end
        end
    end
    return P/(6L^3) # *2?
end

vpressure2(X,F,L) = sum(X.*F)/(3L^3)

temperature(V) = sum(V.^2)/(length(V)/3)   # *m/k se si usano quantità vere


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
    for j = 1:length(A)
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
# don't run this with more than ~1000 frames unless you have a lot of spare time...
function makeVideo(M; T=-1, rho=-1, fps = 30, showCM=true)
    close("all")
    N = Int(size(M,1)/3)
    L = cbrt(N/rho)
    println("\nI'm cooking pngs to make a nice video. It will take some time...")
    prog = Progress(size(M,2), dt=1, barglyphs=BarGlyphs("[=> ]"), barlen=50)  # initialize progress bar

    anim = @animate for i =1:size(M,2)
        Plots.scatter(M[1:3:3N-2,i], M[2:3:3N-1,i], M[3:3:3N,i], m=(10,0.9,:blue,Plots.stroke(0)),w=7, xaxis=("x",(-L/2,L/2)), yaxis=("y",(-L/2,L/2)), zaxis=("z",(-L/2,L/2)), leg=false)
        if showCM   # add center of mass indicator
            cm = avg3D(M[:,i])
            scatter!([cm[1]],[cm[2]],[cm[3]], m=(16,0.9,:red,Plots.stroke(0)))
        end
        next!(prog) # increment the progress bar
    end
    file = string("./3-MD_1/Video/LJ",N,"_T",T,"_d",rho,".mp4")
    mp4(anim, file, fps = fps)
    gui() #show the last frame in a separate window
end

function saveCSV(M; N="???", T="???", rho="???")
    D = convert(DataFrame, M)
    file = string("./3-MD_1/Data/LJ",N,"_T",T,"_d",rho,".csv")
    CSV.write(file, D)
    info("System saved in ", file)
end

function make3Dplot(A::Array{Float64}, rho=-1.0)
    N = Int(length(A)/3)
    if rho == -1.0
        Plots.scatter(A[1:3:3N-2], A[2:3:3N-1], A[3:3:3N], m=(7,0.9,:blue,Plots.stroke(0)),w=7, xaxis=("x"), yaxis=("y"), zaxis=("z"), leg=false)
    else
        L = cbrt(N/rho)
        Plots.scatter(A[1:3:3N-2], A[2:3:3N-1], A[3:3:3N], m=(7,0.9,:blue,Plots.stroke(0)),w=7, xaxis=("x",(-L/2,L/2)), yaxis=("y",(-L/2,L/2)), zaxis=("z",(-L/2,L/2)), leg=false)
    end
    gui()
end

function make2DtemporalPlot(M::Array{Float64,2}, rho)
    N = Int(size(M,1)/3)
    L = cbrt(N/rho)
    Na = round(Int,∛(N/4)) # number of cells per dimension
    a = L / Na  # passo reticolare
    X = M[1:3:3N,1]
    #pick the particles near the plane x=a/4
    I = find(abs.(X -a/4) .< a/4)
    # now the X indices of M in the choosen plane are 3I-2
    scatter(M[3I-1,1], M[3I,1], m=(7,0.7,:red,Plots.stroke(0)),w=7, xaxis=("x",(-L/2,L/2)), yaxis=("y",(-L/2,L/2)), leg=false)
    for i =2:size(M,2)
        scatter!(M[3I+-1,i], M[3I,i], m=(7,0.1,:blue,Plots.stroke(0)))
    end
    gui()
end

function prettyPrint(L, rho, XX, VV, E, T, P, cm)
    M = length(P)
    println("\nPressure: ", mean(P[M÷4:end]), " ± ", std(P[M÷4:end]))
    println("Mean temperature: ", mean(T[M÷4:end]), " ± ", std(T[M÷4:end]))
    println("Mean energy: ", mean(E[M÷4:end]), " ± ", std(E[M÷4:end]))
    println()
end

#XX, EE, TT, PP, CM = simulation(N=108, T0=0.5, rho=0.9, maxsteps=2*10^4, fstep=40, dt=2e-4)
