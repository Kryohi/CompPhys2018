module MC

## TODO
# scelta D in base a media pesata con τ invece che τ migliore, usando vettore D
# recuperare dati da fase di termalizzazione?
# aggiungere check equilibrio con convoluzione per smoothing e derivata discreta
# ...oppure come da appunti
# parallelizzare e ottimizzare
# autocorrelazione in modo più furbo (Wiener–Khinchin?)
# kernel openCL?
# provare a riscrivere in C loop simulazione
# individuare zona di transizione di fase (con cv) con loop su temperature
# implementare reweighting per raffinare picco
# trovare picco per diverse ρ
# grafici
# profit

if VERSION >= v"0.7-"   # su julia master Plots non si compila ¯\_(ツ)_/¯
    using Dates, ProgressMeter
else
    using DataFrames, CSV, ProgressMeter, PyCall, Plots
    pyplot(size=(800, 600))
    fnt = "sans-serif"
    default(titlefont=Plots.font(fnt,24), guidefont=Plots.font(fnt,24), tickfont=Plots.font(fnt,14), legendfont=Plots.font(fnt,14))
end

# add missing directories in current folder
any(x->x=="Data", readdir("./")) || mkdir("Data")
any(x->x=="Plots", readdir("./")) || mkdir("Plots")
any(x->x=="Video", readdir("./")) || mkdir("Video")

# Main function, it creates the initial system, runs a (long) burn-in for thermalization and Δ seclection and then runs a Monte Carlo simulation for maxsteps
function metropolis_ST(; N=256, T=2.0, rho=0.5, Df=1/70, maxsteps=10^5, anim=false)

    Y = zeros(3N)   # array of proposals
    j = zeros(Int64, maxsteps)  # array di frazioni accettate
    U = zeros(Int(maxsteps/fstep)) # array of total energy
    P2 = zeros(U)   # virial pressure

    L = cbrt(N/rho)
    X, a = initializeSystem(N, L)   # creates FCC crystal
    @show D = a*Df    # Δ iniziale lo scegliamo come frazione di passo reticolare
    X, D = burnin(X, D, T, L, a, 120000)  # evolve until at equilibrium, while tuning Δ
    @show D/a; println()

    prog = Progress(maxsteps, dt=1.0, desc="Simulating...", barglyphs=BarGlyphs("[=> ]"), barlen=50)
    @inbounds for n = 1:maxsteps
        P2[i] = vpressure(X,L)
        U[i] = energy(X,L)
        Y .= X .+ D.*(rand(3N).-0.5)    # Proposta
        shiftSystem!(Y,L)
        ap = exp((U[i] - energy(Y,L))/T)   # P[Y]/P[X]
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
    P = P2.+rho*T

    C_H = autocorrelation(H, 1000)   # quando funzionerà sostituire il return con tau
    @show τ = sum(C_H)
    @show CV = cv(H,T,τ)
    @show CVignorante = variance(H[1:200:end])/T^2 + 1.5T

    prettyPrint(T, rho, H, P, CV, τ)
    anim && makeVideo(XX, T=T, rho=rho, D=D)

    return nothing, H, P, j./(3N), C_H, CV, CVignorante
end

# faster(?) version with thermodinamic parameters computed every fstep steps
function metropolis_ST(fstep::Int; N=256, T=2.0, rho=0.5, Df=1/70, maxsteps=10^5, anim=false)

    Y = zeros(3N)   # array of proposals
    j = zeros(Int64, maxsteps)  # array di frazioni accettate
    #XX = zeros(3N, Int(maxsteps/fstep)) # positions history
    U = zeros(Int(maxsteps/fstep)) # array of total energy
    P2 = zeros(U)   # virial pressure

    L = cbrt(N/rho)
    X, a = initializeSystem(N, L)   # creates FCC crystal
    @show D = a*Df    # Δ iniziale lo scegliamo come frazione di passo reticolare
    X, D = burnin(X, D, T, L, a, 120000)  # evolve until at equilibrium, while tuning Δ
    @show D/a; println()

    prog = Progress(maxsteps, dt=1.0, desc="Simulating...", barglyphs=BarGlyphs("[=> ]"), barlen=50)
    @inbounds for n = 1:maxsteps
        if (n-1)%fstep == 0
            i = cld(n,fstep)
            P2[i] = vpressure(X,L)
            U[i] = energy(X,L)
            #XX[:,i] = X
        end
        Y .= X .+ D.*(rand(3N).-0.5)    # Proposta
        shiftSystem!(Y,L)
        ap = exp((energy(X,L) - energy(Y,L))/T)   # P[Y]/P[X]
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
    P = P2.+rho*T

    C_H = autocorrelation(H, 1000)   # quando funzionerà sostituire il return con tau
    @show τ = sum(C_H)
    @show CV = cv(H,T,τ)    # in questo caso inutile e sbagliato
    @show CVignorante = variance(H)/T^2 + 1.5T

    prettyPrint(T, rho, H, P, CV, τ)
    anim && makeVideo(XX, T=T, rho=rho, D=D)

    return nothing, H, P, j./(3N), C_H, CV, CVignorante
end


## WIP: doesn't really work yet
# Multi-threaded implementation, without history of positions and progress bar
function metropolis_MT(; N=500, T=2.0, rho=0.5, Df=1/20, maxsteps=10^4)

    #Y = zeros(3N)   # array of proposals
    j = SharedArray{Int64}(maxsteps)  # array di frazioni accettate
    #XX = zeros(3N, Int(maxsteps/fstep)) # positions history
    U = SharedArray{Float64}(maxsteps) # array of total energy
    P2 = SharedArray{Float64}(maxsteps)   # virial pressure

    L = cbrt(N/rho)
    X, a = initializeSystem(N, L, T)   # creates FCC crystal
    @show D = a*Df    # Δ iniziale lo scegliamo come frazione di passo reticolare
    X, D, jbi = burnin(X, D, T, L, a)  # evolve until at equilibrium, while tuning Δ
    @show D/a
    println()

    Threads.@threads for n = 1:maxsteps
        Y = X .+ D.*(rand(3N).-0.5)    # Proposal
        shiftSystem!(Y,L)
        ap = exp((energy(X,L) - energy(Y,L))/T)   # P[Y]/P[X]
        η = rand(3N)
        for i = 1:length(X)
            if η[i] < ap
                X[i] = Y[i]
                j[n] += 1
            end
        end
        U[n] = energy(X,L)
        P2[n] = vpressure(X,L)
    end

    H = U.+3N*T/2
    CV = cv(H,T)
    prettyPrint(T, rho, H, P2.+rho*T, CV)

    return H, P2.+rho*T, CV, jbi, j./(3N)
end

## WIP: doesn't save arrays (expected) but neither can calculate CV
# multi process implementation, using pmap
function metropolis_MP(; N=500, T=2.0, rho=0.5, Df=1/20, maxsteps=10^4)

    L = cbrt(N/rho)
    X, a = initializeSystem(N, L, T)   # creates FCC crystal
    @show D = a*Df    # Δ iniziale lo scegliamo come frazione di passo reticolare
    X, D, jbi = burnin(X, D, T, L, a)  # evolve until at equilibrium, while tuning Δ
    @show D/a
    println()

    function metropolis(X::Array{Float64}, seed::Int, steps::Int64) # da portare fuori prima o poi
        Y = zeros(3N)   # array of proposals
        E, P2, j = 0.0, 0.0, 0.0
        srand(seed)
        for i = 1:steps
            Y .= X .+ D.*(rand(3N).-0.5)    # Proposal
            shiftSystem!(Y,L)
            ap = exp((energy(X,L) - energy(Y,L))/T)   # P[Y]/P[X]
            η = rand(3N)
            @inbounds for i = 1:length(X)
                if η[i] < ap
                    X[i] = Y[i]
                    j += 1
                end
            end
            E += energy(X,L)
            P2 += vpressure(X,L)
        end
        @show P2/steps
        @show j/steps
        return E/steps, P2/steps, j/steps
    end

    R = pmap(n -> metropolis(X, n, maxsteps÷4), [68 42 1 69]')
    U, P2, j = [x[1] for x in R], [x[2] for x in R], [x[3] for x in R]

    H = mean(U) + 3N*T/2
    P = mean(P2)+rho*T
    CV = 0.0 #cv(H,T)
    prettyPrint(T, rho, H, P, CV)
    return H, P, CV, jbi, mean(j)/3N
end

function metropolis_GPU(; N=500, T=2.0, rho=0.5, Df=1/20, maxsteps=10^4)
end

## -------------------------------------
## Initialization
##

# Initialize the system at t=0 as a perfect FCC crystal centered in 0
function initializeSystem(N::Int, L)
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
    X += (rand(3N) - .5) .* (a/5)
    return X, a
end

# al momento setta solo D, vorremmo che facesse raggiungere anche l'eq termodinamico
# DA RISCRIVERE USANDO 2 LOOP
function burnin(X::Array{Float64}, D::Float64, T::Float64, L::Float64, a::Float64, maxsteps::Int64)

    wnd = maxsteps ÷ 10
    k_max = 750  # distanza per autocorrelazione
    N = Int(length(X)/3)
    j = zeros(maxsteps)
    jm = zeros(maxsteps÷wnd)
    Y = zeros(3N)
    U = zeros(maxsteps)
    H = zeros(maxsteps)
    C_H_tot = []
    τ = ones(maxsteps÷wnd).*1e6
    DD = zeros(maxsteps÷wnd*k_max)    # solo per grafico stupido
    D_chosen = D    # D da restituire, minimizza autocorrelazione

    @inbounds for n=1:maxsteps
        # Proposta
        Y .= X .+ D.*(rand(3N).-0.5)
        shiftSystem!(Y,L)
        # P[Y]/P[X]
        ap = exp((energy(X,L) - energy(Y,L))/T)
        η = rand(3N)
        for i = 1:3N
            if η[i] < ap
                X[i] = Y[i]
                j[n] += 1
            end
        end
        U[n] = energy(X,L)
        H = U.+3N*T/2
        #push!(HH, H)

        # ogni wnd passi calcola autocorrelazione e aggiorna D
        if n%wnd == 0
            DD[(n÷wnd*k_max-k_max+1):n÷wnd*k_max] = D # per garfico stupido
            meanH = mean(H[n-wnd+1:n])
            C_H_temp = zeros(k_max)
            C_H = ones(k_max)

            # da sostituire con correlation() ?
            for k = 1:k_max
                for i = n-wnd+1:n-k_max-1
                    C_H_temp[k] += H[i]*H[i+k-1]    # -meanH a entrambi?
                end
                C_H_temp[k] = C_H_temp[k] / (wnd - k_max)
                C_H[k] = (C_H_temp[k] - meanH^2)/(C_H_temp[1] - meanH^2) # andrà bene l'abs?
            end
            C_H_tot = [C_H_tot; C_H]

            # solo per controllare che cacchio sta succedendo
            @show τ[n÷wnd] = sum(C_H)
            #@show CV = cv(H,T,τ[n÷wnd])

            # check sulla media di passi accettati nella finestra attuale
            @show jm[n÷wnd] = mean(j[(n-wnd+1):n])./(3N)
            if jm[n÷wnd] > 0.2 && jm[n÷wnd] < 0.6
                # if acceptance rate is good, choose D to minimize autocorrelation
                if n>wnd*2 && τ[n÷wnd] < minimum(filter(x->x.>0, τ[1:n÷wnd-1])) && τ[n÷wnd]>0
                    @show D_chosen = D
                end
                @show D = D*(1 + rand()/2 - 0.25)

            elseif jm[n÷wnd] < 0.2
                @show D = D*0.7
                τ[n÷wnd] = 1e6
            else
                @show D = D*1.3
                τ[n÷wnd] = 1e6
            end
        end
    end

    boh = plot(C_H_tot, yaxis=("P",(-1.5,2.5)), linewidth=1.5, leg=false, reuse=false)
    plot!(boh, DD.*30)
    plot!(boh, 1:k_max:(maxsteps÷wnd*k_max), τ./200)
    gui()
    @show D, D_chosen

    plot!(H./H[1])
    return X, D_chosen
end


## -------------------------------------
## Evolution
##

HO(x::Float64,ω::Float64) = ω^2*x^2 /2
LJ(dr::Float64) = 4*(dr^-12 - dr^-6)
der_LJ(dr::Float64) = 4*(6*dr^-8 - 12*dr^-14)   # (dV/dr)/r

function shiftSystem!(A::Array{Float64,1}, L::Float64)
    @inbounds for j = 1:length(A)
        A[j] = A[j] - L*round(A[j]/L)
    end
end

function autocorrelation(H::Array{Float64,1}, k_max::Int64) # return τ when saremo sicuri che funzioni

    meanH = mean(H)
    C_H_temp = zeros(k_max)
    C_H = zeros(k_max)

    for k = 1:k_max
        for i = 1:length(H)-k_max-1
            C_H_temp[k] += H[i]*H[i+k-1]
        end
        C_H_temp[k] = C_H_temp[k] / (length(H)-k_max)
        C_H[k] = (C_H_temp[k] - meanH^2)/(C_H_temp[1] - meanH^2)
    end
    return C_H
    #@show return τ = sum(C_H)
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
                V += LJ(sqrt(dr2))  # cambiare togliendo LJ
            end
        end
    end
    return V
end

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
                P += der_LJ(sqrt(dr2))*dr2  # cambiare togliendo LJ
            end
        end
    end
    return -P/(3L^3)
end


variance(A::Array{Float64}) = mean(A.*A) - mean(A)^2
variance2(A::Array{Float64}, τ) = (mean(A.*A) - mean(A)^2)*τ/length(A)

cv(H::Array{Float64}, T::Float64, τ::Float64) = τ*variance(H)/T^2 + 1.5T


@fastmath function orderParameter(XX, rho)
    N = Int(size(XX,1)/3)
    L = cbrt(N/rho)
    Na = round(Int,∛(N/4)) # number of cells per dimension
    a = L / Na  # passo reticolare
    r = XX[:,size(XX,2)÷3:end]  # taglia parti non all'equilibrio
    dx = zeros(Na^3*3,size(r,2))
    dy = zeros(dx)
    dz = zeros(dx)
    @inbounds for k=0:Na^3-1
        for i=1:3
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


# makes an mp4 video made by a lot of 3D plots (can be easily modified to produce a gif instead)
# don't run this with more than ~1000 frames unless you have a lot of spare time...
# function makeVideo(M; T=-1, rho=-1, fps = 30, D=-1.0, showCM=false)
#     close("all")
#     Plots.default(size=(1280,1080))
#     N = Int(size(M,1)/3)
#     rho==-1 ? L = cbrt(N/(2*maximum(M))) : L = cbrt(N/rho)
#     println("\nI'm cooking pngs to make a nice video. It will take some time...")
#     prog = Progress(size(M,2), dt=1, barglyphs=BarGlyphs("[=> ]"), barlen=50)  # initialize progress bar
#
#     anim = @animate for i =1:size(M,2)
#         Plots.scatter(M[1:3:3N-2,i], M[2:3:3N-1,i], M[3:3:3N,i], m=(10,0.9,:blue,Plots.stroke(0)),w=7, xaxis=("x",(-L/2,L/2)), yaxis=("y",(-L/2,L/2)), zaxis=("z",(-L/2,L/2)), leg=false)
#         if showCM   # add center of mass indicator
#             cm = avg3D(M[:,i])
#             scatter!([cm[1]],[cm[2]],[cm[3]], m=(16,0.9,:red,Plots.stroke(0)))
#         end
#         next!(prog) # increment the progress bar
#     end
#     file = string("./Video/LJ",N,"_T",T,"_d",rho,"_D",D,".mp4")
#     mp4(anim, file, fps = fps)
#     gui() #show the last frame in a separate window
# end

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

function prettyPrint(T::Float64, rho::Float64, E::Array, P::Array, cv::Float64, τ)
    l = length(P)
    println("\nPressure: ", mean(P), " ± ", sqrt(abs(variance2(P,τ))))
    println("Mean energy: ", mean(E), " ± ", std(E))
    println("Specific heat: ", cv)
    println()
end
function prettyPrint(T::Float64, rho::Float64, E::Float64, P::Float64, cv::Float64)
    l = length(P)
    println("\nPressure: ", P, " ± ", 0.0)
    println("Mean energy: ", E, " ± ", 0.0)
    println("Specific heat: ", cv)
    println()
end

function avg3D(A::Array{Float64,1})
    N = Int(length(A)/3)
    return [sum(A[1:3:N-2]), sum(A[2:3:N-1]), sum(A[3:3:N])]./N
end


# creates an array with length N of gaussian distributed numbers using Box-Muller
function vecboxMuller(sigma, N::Int, x0=0.0)
    #srand(60)   # sets the rng seed, to obtain reproducible numbers
    x1 = rand(Int(N/2))
    x2 = rand(Int(N/2))
    @. [sqrt(-2sigma*log(1-x1))*cos(2π*x2); sqrt(-2sigma*log(1-x2))*sin(2π*x1)]
end

end
