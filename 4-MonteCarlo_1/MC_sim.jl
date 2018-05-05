module MC

## TODO
# calcolo giusto di varianza (manca M)
# raffinare scelta D, sia come parametri evolutivi che con metodo varianza
# aggiungere check equilibrium con convoluzione per smoothing e derivata discreta
# parallelizzare e ottimizzare
# kernel openCL?
# provare a riscrivere in C loop simulazione
# individuare zona di transizione di fase con loop su temperature
# implementare reweighting per raffinare picco
# trovare picco per diverse ρ
# grafici
# profit


using ProgressMeter, DataFrames, CSV, Plots
pyplot(size=(800, 600))
fnt = "sans-serif"
default(titlefont=Plots.font(fnt,24), guidefont=Plots.font(fnt,24), tickfont=Plots.font(fnt,14), legendfont=Plots.font(fnt,14))


# add missing directories in current folder
any(x->x=="Data", readdir("./")) || mkdir("Data")
any(x->x=="Plots", readdir("./")) || mkdir("Plots")
any(x->x=="Video", readdir("./")) || mkdir("Video")

# Main function, it creates the initial system, runs it through the vVerlet algorithm for maxsteps,
# saves the positions arrays every fstep iterations, returns and saves it as a csv file
# and optionally creates an animation of the particles (also doable at a later time from the XX output)
function simulation(; N=256, T=2.0, rho=1.3, Df=1/20, fstep=1, maxsteps=10^4, anim=false, csv=true)

    L = cbrt(N/rho)
    Na = cbrt(N/4)
    a = L / Na
    @show D = a*Df    # Δ lo scegliamo come frazione di passo reticolare (per ora)
    X = initializeSystem(N, L, T)
    X, D, jeq = burnin(X, D, T, L)
    @show D/a
    Y = zeros(3N)
    j = zeros(Int64, maxsteps)
    XX = zeros(3N, Int(maxsteps/fstep)) # storia delle posizioni
    U = zeros(Int(maxsteps/fstep)) # array of total energy
    P2 = zeros(U)
    CM = zeros(3*Int(maxsteps/fstep)) # da togliere

    println()

    prog = Progress(maxsteps, dt=1.0, desc="Simulating...", barglyphs=BarGlyphs("[=> ]"), barlen=50)
    @inbounds for n = 1:maxsteps
        if (n-1)%fstep == 0
            i = cld(n,fstep)    # smallest integer larger than or equal to n/fstep
            P2[i] = vpressure(X,L)
            U[i] = energy(X,L)
            CM[3i-2:3i] = avg3D(X)
            XX[:,i] = X
        end
        # Proposta
        Y .= X .+ D.*(rand(3N).-0.5)
        shiftSystem!(Y,L)
        # P[Y]/P[X]
        ap = exp((energy(X,L) - energy(Y,L))/T)
        η = rand(3N)
        for i = 1:length(X)
            if η[i] < ap
                X[i] = Y[i]
                j[n] += 1
            end
        end
        next!(prog)
    end

    H = U.+3N*T/2
    prettyPrint(L, rho, H, P2.+rho*T, CM)
    csv && saveCSV(XX', N=N, T=T, rho=rho)
    anim && makeVideo(XX, T=T, rho=rho, D=D)

    return XX, CM, H, P2.+rho*T, cv(H,T), jeq, j./(3N)
end


## -------------------------------------
## Initialization
##

# Initialize the system at t=0 as a perfect FCC crystal centered in 0
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
    return X
end

# al momento setta solo D
function burnin(X::Array{Float64}, D::Float64, T::Float64, L::Float64)
    eqstepsmax = 2000
    N = Int(length(X)/3)
    Na = cbrt(N/4)
    a = L / Na
    j = zeros(eqstepsmax)
    jm = zeros(eqstepsmax÷50)
    Y = zeros(3N)
    @inbounds for n=1:eqstepsmax
        # Proposta
        Y .= X .+ D.*(rand(3N).-0.5)
        shiftSystem!(Y,L)
        # P[Y]/P[X]
        @show ap = exp((energy(X,L) - energy(Y,L))/T)
        η = rand(3N)
        for i = 1:3N
            if η[i] < ap
                X[i] = Y[i]
                j[n] += 1
            end
        end
        if n%50 == 0
            @show jm[n÷50+1] = mean(j[(n-49):n])./(3N)
            # se la differenza della media di due blocchi è meno di due centesimi
            # della variazione massima di j, equilibrio raggiunto
            #if abs(jm[n÷50+1]-jm[n÷50]) / (maximum(j)-minimum(j[j.>0])) < 0.02
            if jm[n÷50+1] > 0.5 && jm[n÷50+1] < 0.65
                return X, D, j[1:n]     # da sostituire con check equilibrio termodinamico
            elseif jm[n÷50+1] < 0.55
                @show D -= a/100
            else
                @show D += a/100
            end
        end
    end
    warn("It seems equilibrium was not reached")
    return X, D, j./(3N)
end


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

function energy(r,L)
    V = 0.0
    @inbounds for l=0:Int(length(r)/3)-1
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
    return V
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

avgSquares(A::Array{Float64}) = mean(A.*A)
variance(A::Array{Float64}) = (avgSquares(A)-mean(A)^2)/(length(A)-1) # lenghth(A) è da cambiare

cv(H::Array{Float64}, T::Float64) = variance(H)/T^2 + 1.5T


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


# makes an mp4 video made by a lot of 3D plots (can be easily modified to produce a gif instead)
# don't run this with more than ~1000 frames unless you have a lot of spare time...
function makeVideo(M; T=-1, rho=-1, fps = 30, D=-1.0, showCM=false)
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
    file = string("./Video/LJ",N,"_T",T,"_d",rho,"_D",D,".mp4")
    mp4(anim, file, fps = fps)
    gui() #show the last frame in a separate window
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

function prettyPrint(L, rho, E, P, cm)
    l = length(P)
    println("\nPressure: ", mean(P[l÷4:end]), " ± ", std(P[l÷4:end]))
    println("Mean energy: ", mean(E[l÷4:end]), " ± ", std(E[l÷4:end]))
    println("Mean center of mass: [", mean(cm[l÷4:3:end-2]), ", ", mean(cm[l÷4+1:3:end-1]), ", ", mean(cm[l÷4+2:3:end]), "]")
    println()
end

end
