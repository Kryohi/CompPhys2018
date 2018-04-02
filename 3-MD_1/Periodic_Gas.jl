## TODO
# Capire discrepanza T calcolata e inizializzata (è giusto quel /3?)
# controllare mezzi in pressione, energia, temperatura
# velocityverlet! con where{}
# parametro d'ordine
# velocizzare creazione animazione o minacciare maintainer su github di farlo
# aggiungere entropia, energia libera di Gibbs...
# usare Measurements per variabili termodinamiche medie?
# provare Gadfly master
# 3D temporal plot
# 2D temporal plot più esteso

@everywhere using Plots, ProgressMeter, DataFrames, CSV
pyplot()
PyPlot.PyObject(PyPlot.axes3D)
fnt = "sans-serif"
default(titlefont=Plots.font(fnt,24), guidefont=Plots.font(fnt,24), tickfont=Plots.font(fnt,14), legendfont=Plots.font(fnt,14))

# create missing directories in current folder
any(x->x=="Data", readdir("./")) || mkdir("Data")
any(x->x=="Plots", readdir("./")) || mkdir("Plots")
any(x->x=="Video", readdir("./")) || mkdir("Video")

# Main function, it creates the initial system, runs it through the Verlet algorithm for maxsteps,
# saves the positions arrays every fstep iterations, returns and saves it as a csv file
# and optionally creates an animation of the particles (also doable at a later time from the XX output)
@everywhere function simulation(; N=256, T0=4.0, rho=1.3, dt = 1e-4, fstep = 50, maxsteps = 10^4, anim=false, csv=true, onlyP=false)

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
        if (n-1)%fstep == 0
            i = cld(n,fstep)    # "Smallest integer larger than or equal to n/fstep"
            T[i] = temperature(V)
            P[i] = vpressure2(X,F,L) + T[i]*rho
            if !onlyP
                E[i] = energy(X,V,L)
                CM[3i-2:3i] = avg3D(X)
                XX[:,i] = X
            end
        end
        X, V, F = velocityVerlet(X, V, F, L, dt)
        next!(prog)
    end

    prettyPrint(L, rho, E, T, P, CM)
    csv && saveCSV(XX', N=N, T=T0, rho=rho)
    anim && makeVideo(XX, T=T0, rho=rho)

    return XX, E, T, P, CM # returns a matrix with the hystory of positions, energy and pressure arrays
end


## ----------------------------------
## Initialization
##

# Initialize the system at t=0 as a perfect FCC crystal centered in 0, plus adimensional
# maxwell-boltzmann velocities
@everywhere function initializeSystem(N::Int, L, T)
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
    σ = sqrt(T)/3     #  in qualche unità di misura
    V = vecboxMuller(σ,3N)
    @show temperature(V)
    # force the average velocity to 0
    V[1:3:N-2] .-= 3*sum(V[1:3:N-2])/N   # capire perch serve il 3 e perchè T cambia
    V[2:3:N-1] .-= 3*sum(V[2:3:N-1])/N
    V[3:3:N] .-= 3*sum(V[3:3:N])/N
    #@show [sum(V[1:3:N-2]), sum(V[2:3:N-1]), sum(V[3:3:N])]
    #@show temperature(V)
    return [X, V]
end

# creates an array with length N of gaussian distributed numbers, with σ = sigma
@everywhere function vecboxMuller(sigma, N::Int, x0=0.0)
    srand(60)   # sets the rng seed, to obtain reproducible numbers
    x1 = rand(Int(N/2))
    x2 = rand(Int(N/2))
    @. [sqrt(-2sigma*log(1-x1))*cos(2π*x2); sqrt(-2sigma*log(1-x2))*sin(2π*x1)]
end

@everywhere function shiftSystem!(A::Array{Float64,1}, L::Float64)
    for j = 1:length(A)
        @inbounds A[j] = A[j] - L*round(A[j]/L)
    end
    nothing
end


## ----------------------------------
## Evolution
##

@everywhere LJ(dr::Float64) = 4*(dr^-12 - dr^-6)
@everywhere der_LJ(dr::Float64) = 4*(6*dr^-8 - 12*dr^-14)

@everywhere function forces(r::Array{Float64,1}, L::Float64)
    F = zeros(r)
    # multithreading is convenient only for large N
    # to set the number of threads use the environment variable JULIA_NUM_THREADS=4, or go to settings in Atom
    for l=0:Int(length(r)/3)-1
        @inbounds @simd for i=0:l-1
            dx = r[3l+1] - r[3i+1]
            dx = dx - L*round(dx/L)
            dy = r[3l+2] - r[3i+2]
            dy = dy - L*round(dy/L)
            dz = r[3l+3] - r[3i+3]
            dz = dz - L*round(dz/L)
            dr2 = dx*dx + dy*dy + dz*dz
            if dr2 < L*L/4
                #dV = -der_LJ(sqrt(dr2))
                dV = -24*dr2^-4 + 48*dr2^-7
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


@everywhere @inbounds function velocityVerlet(x, v, F, L, dt)
    @. x += v*dt + F*dt^2/2
    shiftSystem!(x, L)
    F_ = forces(x,L)
    @. v += (F + F_)*dt/2
    return x, v, F_
end


## ----------------------------------
## Thermodinamic Properties
##

@everywhere function energy(r,v,L)
    T = (v[1]^2 + v[2]^2 + v[3]^2)/2 #perché nel ciclo sotto il primo elemento non verrebbe considerato
    V = 0.0
    @inbounds for l=1:Int(length(r)/3)-1
        T += (v[3l+1]^2 + v[3l+2]^2 + v[3l+3]^2)./2
        @simd for i=0:l-1
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

@everywhere @fastmath @inbounds temperature(V) = sum(V.^2)/(length(V)/3)   # *m/k se si usano quantità vere

@everywhere @fastmath @inbounds vpressure2(X,F,L) = sum(X.*F)/(3L^3)

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

function avg3D(A::Array{Float64,1})
    N = Int(length(A)/3)
    return [sum(A[1:3:N-2]), sum(A[2:3:N-1]), sum(A[3:3:N])]./N
end

# fare attenzione a non prendere particella vicino ai bordi
function lindemann(X0, XX, N, rho)   # where X0 is a triplet at t=0, XX the hystory of that point at t>0
    L = cbrt(N/rho)
    Na = round(Int,∛(N/4)) # number of cells per dimension
    a = L / Na  # passo reticolare
    deltaX = sqrt(sum((XX[1,:]-X0[1]).^2 .+ (XX[2,:]-X0[2]).^2 .+ (XX[3,:]-X0[3]).^2) / (length(XX[1,:])-1))
    return deltaX*2/a
end


## ----------------------------------
## Visualization
##

# makes an mp4 video made by a lot of 3D plots (can be easily modified to produce a gif instead)
# don't run this with more than ~1000 frames unless you have a lot of spare time...
function makeVideo(M; T=-1, rho=-1, fps = 30, showCM=false)
    close("all")
    Plots.default(size=(1280,1080))
    N = Int(size(M,1)/3)
    rho==-1 ? L = cbrt(N/(2*maximum(M))) : L = cbrt(N/rho)
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
    file = string("./Video/LJ",N,"_T",T,"_d",rho,".mp4")
    mp4(anim, file, fps = fps)
    gui() #show the last frame in a separate window
end

function make3Dplot(A::Array{Float64}, T= -1.0, rho=-1.0)
    Plots.default(size=(800,600))
    N = Int(length(A)/3)
    if rho == -1.0
        Plots.scatter(A[1:3:3N-2], A[2:3:3N-1], A[3:3:3N], m=(7,0.9,:blue,Plots.stroke(0)),w=7, xaxis=("x"), yaxis=("y"), zaxis=("z"), leg=false)
    else
        L = cbrt(N/rho)
        Plots.scatter(A[1:3:3N-2], A[2:3:3N-1], A[3:3:3N], m=(7,0.9,:blue,Plots.stroke(0)),w=7, xaxis=("x",(-L/2,L/2)), yaxis=("y",(-L/2,L/2)), zaxis=("z",(-L/2,L/2)), leg=false)
    end
    gui()
end

@everywhere function make2DtemporalPlot(M::Array{Float64,2}; T=-1.0, rho=-1.0, save=true)
    Plots.default(size=(800,600))
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
        scatter!(M[3I+-1,i], M[3I,i], m=(7,0.05,:blue,Plots.stroke(0)))
    end
    file = string("./Plots/temporal2D_",N,"_T",T,"_d",rho,".pdf")
    save && savefig(file)
    gui()
end


## ----------------------------------
## Miscellaneous
##

@everywhere function saveCSV(M; N="???", T="???", rho="???")
    D = convert(DataFrame, M)
    file = string("./Data/positions_",N,"_T",T,"_d",rho,".csv")
    CSV.write(file, D)
    info("System saved in ", file)
end

@everywhere function prettyPrint(L, rho, E, T, P, cm)
    l = length(P)
    println("\nPressure: ", mean(P[l÷3:end]), " ± ", std(P[l÷3:end])/sqrt(l*2/3))
    println("Mean temperature: ", mean(T[l÷3:end]), " ± ", std(T[l÷3:end])/sqrt(l*2/3))
    println("Mean energy: ", mean(E[l÷3:end]), " ± ", std(E[l÷3:end])/sqrt(l*2/3))
    println("Mean center of mass: [", mean(cm[l÷3:3:end-2]), ", ", mean(cm[l÷3+1:3:end-1]), ", ", mean(cm[l÷3+2:3:end]))
    println()
end


# @time XX, EE, TT, PP, CV = simulation(N=256, T0=0.5, rho=1.5, maxsteps=1*10^5, fstep=100, dt=5e-4, anim=false)
#
# plot(CV[1:3:end-2])
# gui()
