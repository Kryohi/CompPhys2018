module MC

## TODO
# sostituire check equilibrium con convoluzione per smoothing e derivata discreta

using ProgressMeter, DataFrames, CSV, Plots

# add missing directories in current folder
any(x->x=="Data", readdir("./")) || mkdir("Data")
any(x->x=="Plots", readdir("./")) || mkdir("Plots")
any(x->x=="Video", readdir("./")) || mkdir("Video")


function simulation(; N=500, D=0.5, T0=3.0, maxsteps=10^4)

    c = 1/D # non ho ancora capito dove va usato
    # inizializzazione come gaussiana, ma potrebbe anche essere uniforme
    X = vecboxMuller(1.0,3N)
    Y = zeros(3N)
    j = zeros(Int64, maxsteps)
    X, jeq = equilibrium(X,D,T0)

    for n=1:maxsteps
        # Proposta
        Y .= X .+ D.*(rand(3N).-0.5)
        #shiftSystem!(Y,10.0) # serve? boh
        # P[Y]/P[X]
        ap = exp.((HO.(X,.5) - HO.(Y,.5))/T0)
        η = rand(3N)
        for i = 1:length(X)
            if η[i] < ap[i]
                X[i] = Y[i]
                j[n] += 1
            end
        end
    end
    return X, jeq, j./(3N)
end


## -------------------------------------
## Initialization
##

# Initialize the system at t=0 as a perfect FCC crystal centered in 0, plus adimensional
# maxwell-boltzmann velocities
# suppongo non serva più?
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
    #@show [sum(V[1:3:N-2]), sum(V[2:3:N-1]), sum(V[3:3:N])]./N
    # force the average velocity to 0
    V[1:3:N-2] .-= 3*sum(V[1:3:N-2])/N   # capire perch serve il 3 e perchè T cambia
    V[2:3:N-1] .-= 3*sum(V[2:3:N-1])/N
    V[3:3:N] .-= 3*sum(V[3:3:N])/N
    #@show [sum(V[1:3:N-2]), sum(V[2:3:N-1]), sum(V[3:3:N])]./N
    @show avg3D(X)
    #@show temperature(V)
    return [X, V]
end

function equilibrium(X, D, T0)
    eqstepsmax = 2000
    N = Int(length(X)/3)
    j = zeros(eqstepsmax)
    jm = zeros(eqstepsmax÷50)
    Y = zeros(3N)
    for n=1:eqstepsmax
        # Proposta
        Y .= X .+ D.*(rand(3N).-0.5)
        # P[Y]/P[X]
        ap = exp.((HO.(X,.5) - HO.(Y,.5))/T0)
        η = rand(3N)
        for i = 1:3N
            if η[i] < ap[i]
                X[i] = Y[i]
                j[n] += 1
            end
        end
        if n%50 == 0
            jm[n÷50+1] = mean(j[(n-49):n])
            # se la differenza della media di due blocchi è meno di due centesimi
            # della variazione massima di j, equilibrio raggiunto
            if abs(jm[n÷50+1]-jm[n÷50]) / (maximum(j)-minimum(j[j.>0])) < 0.02
                return X, j[1:n]./(3N)
            end
        end
    end
    warn("It seems equilibrium was not reached")
    return X, j./(3N)
end

Tgauss(x,y) = exp(-(x-y)^2/2)/√(2π)

# creates an array with length N of gaussian distributed numbers using Box-Muller
function vecboxMuller(sigma, N::Int, x0=0.0)
    #srand(60)   # sets the rng seed, to obtain reproducible numbers
    x1 = rand(Int(N/2))
    x2 = rand(Int(N/2))
    @. [sqrt(-2sigma*log(1-x1))*cos(2π*x2); sqrt(-2sigma*log(1-x2))*sin(2π*x1)]
end

function shiftSystem!(A::Array{Float64,1}, L::Float64)
    @inbounds for j = 1:length(A)
        A[j] = A[j] - L*round(A[j]/L)
    end
end


## -------------------------------------
## Evolution
##

HO(x::Float64,ω::Float64) = ω^2*x^2 /2
LJ(dr::Float64) = 4*(dr^-12 - dr^-6)
der_LJ(dr::Float64) = 4*(6*dr^-8 - 12*dr^-14)   # (dV/dr)/r

function metropolis(x, D)

end


## -------------------------------------
## Thermodinamic Properties
##

function energy(r,v,L)
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

@fastmath @inbounds temperature(V) = sum(V.^2)/(length(V)/3)   # *m/k se si usano quantità vere

@fastmath @inbounds vpressure2(X,F,L) = sum(X.*F)/(3L^3)    # non ultraortodosso ma più veloce

@fastmath function vpressure(r,L)
    P = 0.0
    @inbounds for l=1:Int(length(r)/3)-1
        for i=0:l-1
            dx = r[3l+1] - r[3i+1]
            dx = dx - L*round(dx/L)
            dy = r[3l+2] - r[3i+2]
            dy = dy - L*round(dy/L)
            dz = r[3l+3] - r[3i+3]
            dz = dz - L*round(dz/L)
            dr2 = dx^2 + dy^2 + dz^2
            if dr2 < L*L/4
                P += der_LJ(sqrt(dr2))*dr2
            end
        end
    end
    return -P/(3L^3)
end

function avg3D(A::Array{Float64,1})
    N = Int(length(A)/3)
    return [sum(A[1:3:N-2]), sum(A[2:3:N-1]), sum(A[3:3:N])]./N
end


function orderParameter(XX, rho)
    N = Int(size(XX,1)/3)
    L = cbrt(N/rho)
    Na = round(Int,∛(N/4)) # number of cells per dimension
    a = L / Na  # passo reticolare
    r = XX[:,size(XX,2)÷3:end]  # taglia parti non all'equilibrio
    dx = zeros(Na^3*3,size(r,2))
    dy = zeros(dx)
    dz = zeros(dx)
    for k=0:Na^3-1
        @inbounds for i=1:3
            dx[3k+i,:] = r[12k+1,:] - r[12k+3i+1,:]
            dx[3k+i,:] .-= L.*round.(dx[3k+i,:]/L)
            dy[3k+i,:] = r[12k+2,:] - r[12k+3i+2,:]
            dy[3k+i,:] .-= L.*round.(dy[3k+i,:]/L)
            dz[3k+i,:] = r[12k+3,:] - r[12k+3i+3,:]
            dz[3k+i,:] .-= L.*round.(dz[3k+i,:]/L)
        end
    end
    dr = sqrt.(dx.^2 + dy.^2 + dz.^2)
    R = dr[:,1]
    K = 2π./R
    ordPar = mean((cos.(K.*dr)),2)
    return mean(ordPar)
end

## -------------------------------------
## Visualization
##

function make3Dplot(A::Array{Float64}; T= -1.0, rho=-1.0)
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

function make2DtemporalPlot(M::Array{Float64,2}; T=-1.0, rho=-1.0, save=true)
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
        scatter!(M[3I-1,i], M[3I,i], m=(7,0.05,:blue,Plots.stroke(0)), markeralpha=0.05)
    end
    file = string("./Plots/temporal2D_",N,"_T",T,"_d",rho,".pdf")
    save && savefig(file)
    gui()
end


## -------------------------------------
## Miscellaneous
##

function saveCSV(M; N="???", T="???", rho="???")
    D = convert(DataFrame, M)
    file = string("./Data/positions_",N,"_T",T,"_d",rho,".csv")
    CSV.write(file, D)
    info("System saved in ", file)
end

function prettyPrint(L, rho, E, T, P, cm)
    l = length(P)
    println("\nPressure: ", mean(P[l÷3:end]), " ± ", std(P[l÷3:end])/sqrt(l*2/3))
    println("Mean temperature: ", mean(T[l÷3:end]), " ± ", std(T[l÷3:end])/sqrt(l*2/3))
    println("Mean energy: ", mean(E[l÷3:end]), " ± ", std(E[l÷3:end])/sqrt(l*2/3))
    println("Mean center of mass: [", mean(cm[l÷3:3:end-2]), ", ", mean(cm[l÷3+1:3:end-1]), ", ", mean(cm[l÷3+2:3:end]), "]")
    println()
end

end
