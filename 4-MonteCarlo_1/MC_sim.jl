module MC

## TODO
# scelta D in base a media pesata con τ invece che τ migliore, usando vettore D
# aumentare raggio di autocorrelazione per scelta di D
# recuperare dati da fase di termalizzazione?
# limitare calcoli per pressione, che non è così importante
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
    default(titlefont=Plots.font(fnt,24), guidefont=Plots.font(fnt,24), tickfont=Plots.font(fnt,14),
     legendfont=Plots.font(fnt,14))
end

# add missing directories in current folder
any(x->x=="Data", readdir("./")) || mkdir("Data")
any(x->x=="Plots", readdir("./")) || mkdir("Plots")
any(x->x=="Video", readdir("./")) || mkdir("Video")

# Main function, it creates the initial system, runs a (long) burn-in for thermalization
# and Δ selection and then runs a Monte Carlo simulation for maxsteps
function metropolis_ST(; N=108, T=2.0, rho=0.5, Df=1/70, maxsteps=10^6, bmaxsteps=12*10^5, anim=false)

    Y = zeros(3N)   # array of proposals
    j = zeros(Int64, maxsteps)  # array di frazioni accettate
    U = zeros(Float64, maxsteps)    # array of total energy
    P2 = zeros(U)   # virial pressure

    L = cbrt(N/rho)
    X, a = initializeSystem(N, L)   # creates FCC crystal
    D = a*Df    # Δ iniziale lo scegliamo come frazione di passo reticolare
    X, D = burnin(X, D, T, L, a, bmaxsteps)  # evolve until at equilibrium, while tuning Δ
    @show D/a; println()

    prog = Progress(maxsteps, dt=1.0, desc="Simulating...", barglyphs=BarGlyphs("[=> ]"), barlen=50)
    @inbounds for n = 1:maxsteps
        U[n] = energy(X,L)
        P2[n] = vpressure(X,L)
        Y .= X .+ D.*(rand(3N).-0.5)    # Proposta
        shiftSystem!(Y,L)
        ap = exp((U[n] - energy(Y,L))/T)   # P[Y]/P[X]
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

    C_H = autocorrelation(H, 30000)   # quando funzionerà sostituire il return con tau
    τ = sum(C_H)
    CV = cv(H,T,C_H)
    CV2 = variance(H[1:ceil(Int,τ/5):end])/T^2 + 1.5T
    prettyPrint(T, rho, H, P, τ, CV, CV2)
    ##anim && makeVideo(XX, T=T, rho=rho, D=D)

    return H, P, j./(3N), C_H, CV, CV2
end

# faster(?) version with thermodinamic parameters computed every fstep steps
# obviously cannot use τ
# May be utterly useless
function metropolis_ST(fstep::Int; N=108, T=2.0, rho=0.5, Df=1/70, maxsteps=10^5, bmaxsteps=12*10^5, anim=false)

    info("using slim simulation with fstep = ", fstep)
    Y = zeros(3N)   # array of proposals
    j = zeros(Int64, maxsteps)  # array di frazioni accettate
    #XX = zeros(3N, Int(maxsteps/fstep)) # positions history
    U = zeros(Int(maxsteps/fstep))  # array of total energy
    P2 = zeros(U)   # virial pressure

    L = cbrt(N/rho)
    X, a = initializeSystem(N, L)   # creates FCC crystal
    @show D = a*Df    # Δ iniziale lo scegliamo come frazione di passo reticolare
    X, D = burnin(X, D, T, L, a, bmaxsteps)  # evolve until at equilibrium, while tuning Δ
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

    C_H = autocorrelation(H, 10000)   # quando funzionerà sostituire il return con tau
    τ = sum(C_H)
    CV = cv(H,T,τ)    # in questo caso inutile e sbagliato
    CVignorante = variance(H)/T^2 + 1.5T
    #op = Sim.orderParameter(XX, rho)
    #Sim.make2DtemporalPlot(XX[:,1:1700], T=T0, rho=rho, save=true)
    prettyPrint(T, rho, H, P, τ, CV, CVignorante)

    return H, P, j./(3N), C_H, CV, CVignorante
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
    X += (rand(3N) - .5) .* (a/42)
    return X, a
end

# al momento setta solo D, vorremmo che facesse raggiungere anche l'eq termodinamico
# DA RISCRIVERE USANDO 2 LOOP
function burnin(X::Array{Float64}, D0::Float64, T::Float64, L::Float64, a::Float64, maxsteps::Int64)

    wnd = maxsteps ÷ 15 # larghezza finestra su cui fissare D e calcolare tau
    k_max = wnd ÷ 10  # distanza massima per autocorrelazione
    N = Int(length(X)/3)
    j = zeros(maxsteps)
    jm = zeros(maxsteps÷wnd)
    Y = zeros(3N)
    U = zeros(maxsteps)
    H = zeros(maxsteps)
    C_H_tot = []
    τ = ones(maxsteps÷wnd).*1e6
    DD = zeros(maxsteps÷wnd*k_max)    # solo per grafico stupido
    D_chosen = D0    # D da restituire, minimizza autocorrelazione
    D = D0

    if N>10 # if using more particles, add some pre-thermalization
        @inbounds for n = 1:10000
            V = energy(X,L)
            Y .= X .+ D.*(rand(3N).-0.5)    # Proposta
            shiftSystem!(Y,L)
            ap = exp((V - energy(Y,L))/T)   # P[Y]/P[X]
            η = rand(3N)
            for i = 1:length(X)
                if η[i] < ap
                    X[i] = Y[i]
                    j[n] += 1
                end
            end
        end
    end

    @inbounds for n=1:maxsteps
        U[n] = energy(X,L)
        # Proposta
        Y .= X .+ D.*(rand(3N).-0.5)
        shiftSystem!(Y,L)
        # P[Y]/P[X]
        ap = exp((U[n] - energy(Y,L))/T)
        η = rand(3N)
        for i = 1:3N
            if η[i] < ap
                X[i] = Y[i]
                j[n] += 1
            end
        end
        H[n] = U[n]+3N*T/2
        #push!(HH, H)

        # ogni wnd passi calcola autocorrelazione e aggiorna D
        if n%wnd == 0
            DD[(n÷wnd*k_max-k_max+1):n÷wnd*k_max] = D # per grafico stupido
            meanH = mean(H[n-wnd+1:n])
            #C_H_temp = zeros(k_max)
            #C_H = ones(k_max)

            C_H = autocorrelation(H[n-wnd+1:n], k_max)
            # da sostituire con correlation() ?
            # for k = 1:k_max
            #     for i = n-wnd+1:n-k_max-1
            #         C_H_temp[k] += (H[i]-meanH) * (H[i+k-1]-meanH)
            #     end
            #     C_H_temp[k] = C_H_temp[k] / (wnd - k_max)
            #     C_H[k] = C_H_temp[k] / C_H_temp[1]
            # end
            C_H_tot = [C_H_tot; C_H]
            plot(C_H)

            # solo per controllare che cacchio sta succedendo
            @show τ[n÷wnd] = sum(C_H)
            #@show CV = cv(H,T,τ[n÷wnd])

            # check sulla media di passi accettati nella finestra attuale
            @show jm[n÷wnd] = mean(j[(n-wnd+1):n])./(3N)
            if jm[n÷wnd] > 0.25 && jm[n÷wnd] < 0.7
                # if acceptance rate is good, choose D to minimize autocorrelation
                # the first condition excludes the τ values found in the first 3 windows,
                # since equilibrium has not been reached yet.
                if n>wnd*3 && τ[n÷wnd]>0 &&
                    (length(filter(x->x.>0, τ[1:n÷wnd-1]))==0 || τ[n÷wnd] < minimum(filter(x->x.>0, τ[1:n÷wnd-1])))
                    @show D_chosen = D
                end
                @show D = D_chosen*(1 + rand()/2 - 0.25)

            elseif jm[n÷wnd] < 0.25
                @show D = D*0.7
                τ[n÷wnd] = 1e6
            else
                @show D = D*1.3
                τ[n÷wnd] = 1e6
            end
        end
    end

    D_chosen == D && warn("No suitable Δ value was found, using default...")

    boh = plot(C_H_tot, yaxis=("cose",(-1.0,2.7)), linewidth=1.5, label="autocorrelation")
    plot!(boh, DD.*30, label="Δ*30")
    plot!(boh, 1:k_max:(maxsteps÷wnd*k_max), τ./1000, label="τ/1000")
    plot!(H[1:10:end]./H[1], label="E/E[1]")
    gui()
    @show D, D_chosen

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

# Da velocizzare
function autocorrelation(H::Array{Float64,1}, k_max::Int64) # return τ when saremo sicuri che funzioni

    meanH = mean(H)
    C_H = zeros(k_max)

    CH1 = 0.0
    @inbounds for i = 1:length(H)-k_max-1
        CH1 += (H[i]-meanH) * (H[i]-meanH)
    end
    CH1 = CH1 / (length(H)-k_max)

    bar = Progress(k_max, dt=1.0, desc="Calculating autocorrelation...", barglyphs=BarGlyphs("[=> ]"), barlen=33)
    @inbounds for k = 2:k_max
        C_H_temp = 0.0
        @fastmath for i = 1:length(H)-k_max-1
            C_H_temp += (H[i]-meanH) * (H[i+k-1]-meanH)
        end
        C_H_temp = C_H_temp / (length(H)-k_max)
        C_H[k] = C_H_temp / CH1
        next!(bar)
    end
    return C_H
end


## -------------------------------------
## Thermodinamic Properties
##

function energy(r::Array{Float64,1},L::Float64)
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
                #V += LJ(sqrt(dr2))
                V += 4*(dr2^-6 - dr2^-3)
            end
        end
    end
    return V
end

@fastmath function vpressure(r::Array{Float64,1},L::Float64)
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
variance2(A::Array{Float64}, ch) =
(mean(A.*A) - mean(A)^2)*(1-sum((1.-(1:length(ch))./length(ch)).*ch)*2/length(A))

cv(H::Array{Float64}, T::Float64, ch::Array{Float64}) = variance2(H,ch)/T^2 + 1.5T


@fastmath function orderParameter(XX, rho::Float64)
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
        Plots.scatter(A[1:3:3N-2], A[2:3:3N-1], A[3:3:3N], m=(7,0.9,:blue,Plots.stroke(0)),w=7,
         xaxis=("x"), yaxis=("y"), zaxis=("z"), leg=false)
    else
        L = cbrt(N/rho)
        Plots.scatter(A[1:3:3N-2], A[2:3:3N-1], A[3:3:3N], m=(7,0.9,:blue,Plots.stroke(0)),w=7,
         xaxis=("x",(-L/2,L/2)), yaxis=("y",(-L/2,L/2)), zaxis=("z",(-L/2,L/2)), leg=false)
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
#         Plots.scatter(M[1:3:3N-2,i], M[2:3:3N-1,i], M[3:3:3N,i], m=(10,0.9,:blue,Plots.stroke(0)),w=7,
# xaxis=("x",(-L/2,L/2)), yaxis=("y",(-L/2,L/2)), zaxis=("z",(-L/2,L/2)), leg=false)
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
    scatter(M[3I-1,1], M[3I,1], m=(7,0.7,:red,Plots.stroke(0)),w=7, xaxis=("x",(-L/2,L/2)),
     yaxis=("y",(-L/2,L/2)), leg=false)
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

function prettyPrint(T::Float64, rho::Float64, E::Array, P::Array, τ, cv, cv2)
    l = length(P)
    println("\nPressure: ", mean(P), " ± ", sqrt(abs(variance2(P,τ))))
    println("Mean energy: ", mean(E), " ± ", std(E))
    println("Specific heat: ", cv)
    println("Specific heat (approximate) : ", cv2)
    println("Average autocorrelation time: ", τ)
    println()
end
function prettyPrint(T::Float64, rho::Float64, E::Float64, P::Float64, cv, cv2)
    l = length(P)
    println("\nPressure: ", P, " ± ", 0.0)
    println("Mean energy: ", E, " ± ", 0.0)
    println("Specific heat: ", cv)
    println("Specific heat (approximate) : ", cv2)
    println("Average autocorrelation time: ", τ)
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
