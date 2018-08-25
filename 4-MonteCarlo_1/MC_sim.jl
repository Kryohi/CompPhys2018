module MC

## TODO
# scelta D in base a media pesata con τ invece che τ migliore, usando vettore D
# recuperare dati da fase di termalizzazione?
# salvare in dataframe D ottimali, da recuperare in simulazioni successive
# limitare calcoli per pressione, che non è così importante
# aggiungere check equilibrio con convoluzione per smoothing e derivata discreta
# sistemare/verificare parametro d'ordine
# parallelizzare e ottimizzare
# ricontrollare fft_acf
# kernel openCL?
# individuare zona di transizione di fase (con cv) con loop su temperature
# trovare picco per diverse ρ
# grafici
# riscrivere in C+OpenMPI

(VERSION >= v"0.7-") && (using Statistics, FFTW, Distributed)
using DataFrames, CSV, ProgressMeter, PyCall, Plots
pyplot(size=(800, 600))
fnt = "sans-serif"
default(titlefont=Plots.font(fnt,24), guidefont=Plots.font(fnt,24), tickfont=Plots.font(fnt,14),
legendfont=Plots.font(fnt,14))

# add missing directories in current folder
any(x->x=="Data", readdir("./")) || mkdir("Data")
any(x->x=="Plots", readdir("./")) || mkdir("Plots")
any(x->x=="Video", readdir("./")) || mkdir("Video")

# Main function, it creates the initial system, runs a (long) burn-in for thermalization
# and Δ selection and then runs a Monte Carlo simulation for maxsteps
function metropolis_ST(; N=108, T=.5, rho=.5, Df=1/70, maxsteps=10^6, bmaxs=18*10^5, csv=false, anim=false)

    Y = zeros(3N)   # array of proposals
    j = zeros(Int64, maxsteps)  # array di frazioni accettate
    U = zeros(Float64, maxsteps)    # array of total energy
    fstep = 200
    P2 = zeros(Int(maxsteps/fstep))   # virial pressure
    OP = zeros(Int(maxsteps/fstep))  # order parameter

    L = cbrt(N/rho)
    X, a = initializeSystem(N, L)   # creates FCC crystal
    D = a*Df    # initial Δ is passed as a fraction of lattice step a
    X, D = burnin(X, D, T, L, a, bmaxs)  # evolve until at equilibrium, while tuning Δ
    @show a/D; println()

    prog = Progress(maxsteps, dt=1.0, desc="Simulating...", barglyphs=BarGlyphs("[=> ]"), barlen=50)
    @inbounds for n = 1:maxsteps
        if (n-1)%fstep == 0
            i = cld(n,fstep)
            P2[i] = vpressure(X,L)
            OP[i] = orderParameter(X,L)
            #XX[:,i] = X
        end
        U[n] = energy(X,L)
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

    @time C_H = fft_acf(H, 36000)   # don't put more than ~36k if using non-fft acf
    τ = sum(C_H)
    CV = cv(H,T,C_H)
    CV2 = variance(H[1:ceil(Int,τ/5):end])/T^2 + 1.5T
    prettyPrint(T, rho, H, P, C_H, CV, CV2, OP)
    csv && saveCSV(rho, N, T, H, P, CV, CV2, C_H, OP)

    return H, P, j./(3N), C_H, CV, CV2, OP
end


## WIP: doesn't save arrays (expected) but neither can calculate CV
# multi process implementation, using pmap
function metropolis_MP(; N=108, T=.4, rho=.2, Df=1/70, maxsteps=10^6, bmaxs=18*10^5)

    L = cbrt(N/rho)
    X, a = initializeSystem(N, L, T)   # creates FCC crystal
    D = a*Df    # initial Δ is passed as a fraction of lattice step a
    X, D = burnin(X, D, T, L, a, bmaxs)  # evolve until at equilibrium, while tuning Δ
    @show a/D; println()


    function metropolis(X::Array{Float64}, seed::Int, steps::Int64) # da portare fuori prima o poi

    end

    R = pmap(n -> metropolis(X, n, maxsteps÷4), [68 42 1 69]')
    U, P2, j = [x[1] for x in R], [x[2] for x in R], [x[3] for x in R]

    H = mean(U) + 3N*T/2
    P = mean(P2)+rho*T
    @time C_H = fft_acf(H, 36000)
    τ = sum(C_H)
    CV = cv(H,T,C_H)
    CV2 = variance(H[1:ceil(Int,τ/5):end])/T^2 + 1.5T
    prettyPrint(T, rho, H, P, C_H, CV, CV2, OP)
    csv && saveCSV(rho, N, T, H, P, CV, CV2, C_H, OP)

    return H, P, j./(3N), C_H, CV, CV2, OP
end


## -------------------------------------
## Initialization
##

# Initialize the system at t=0 as a perfect FCC crystal centered in 0
function initializeSystem(N::Int, L)
    Na = round(Int,cbrt(N/4)) # number of cells per dimension
    a = L / Na  # passo reticolare
    !isapprox(Na, cbrt(N/4)) && error("Can't make a cubic FCC crystal with this N :(")

    X = zeros(Float64, 3N)
    for i=0:Na-1, j=0:Na-1, k = 0:Na-1  # loop over every cell of the cfc lattice
        n = i*Na*Na + j*Na + k # unique number for each triplet i,j,k
        X[n*12+1], X[n*12+2], X[n*12+3] = a*i, a*j, a*k # vertice celle [x1,y1,z1,...]
        X[n*12+4], X[n*12+5], X[n*12+6] = a*i + a/2, a*j + a/2, a*k
        X[n*12+7], X[n*12+8], X[n*12+9] = a*i + a/2, a*j, a*k + a/2
        X[n*12+10], X[n*12+11], X[n*12+12] = a*i, a*j + a/2, a*k + a/2
    end
    X .+= a/4   # needed to avoid particles exactly at the edges of the box
    shiftSystem!(X,L)
    return X, a
end

# bisognerebbe controllare meglio quando si è raggiunto l'eq termodinamico
function burnin(X::Array{Float64}, D0::Float64, T::Float64, L::Float64, a::Float64, maxsteps::Int64)

    wnd = maxsteps ÷ 12 # larghezza finestra su cui fissare D e calcolare tau
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

    # pre-thermalization
    @inbounds for n = 1:50000
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

    # Δ selection + moar thermalization
    @inbounds for n=1:maxsteps
        U[n] = energy(X,L)
        Y .= X .+ D.*(rand(3N).-0.5)
        shiftSystem!(Y,L)
        ap = exp((U[n] - energy(Y,L))/T)
        η = rand(3N)
        for i = 1:3N
            if η[i] < ap
                X[i] = Y[i]
                j[n] += 1
            end
        end
        H[n] = U[n]+3N*T/2

        # ogni wnd passi calcola autocorrelazione e aggiorna D in base ad acceptance ratio e τ
        if n%wnd == 0
            DD[Int(n÷wnd*k_max-k_max+1):Int(n÷wnd*k_max)] .= D # per grafico stupido

            C_H = fft_acf(H[n-wnd+1:n], k_max)  # autocorrelation function in current window
            C_H_tot = [C_H_tot; C_H]
            τ[n÷wnd] = sum(C_H)

            # average acceptance ratio in current window
            jm[n÷wnd] = mean(j[(n-wnd+1):n])./(3N)
            println("\nAcceptance ratio = ", round(jm[n÷wnd]*1e4)/1e4, ",\t τ = ", round(τ[n÷wnd]*1e4)/1e4)

            if jm[n÷wnd] > 0.25 && jm[n÷wnd] < 0.69
                # if acceptance rate is good, choose D to minimize autocorrelation
                # the first condition excludes the τ values found in the first 3 windows,
                # since equilibrium has not been reached yet (probably).
                if n>wnd*3 && τ[n÷wnd] < minimum(abs.(τ[3:n÷wnd-1]))
                    @show D_chosen = D
                end
                @show D = D_chosen*(1 + rand()/2 - 0.25)

            elseif jm[n÷wnd] < 0.25
                @show D = D*0.7
                τ[n÷wnd] = 1e6  # big value, so it won't get chosen
            else
                @show D = D*1.3
                τ[n÷wnd] = 1e6
            end
        end
    end

    if D_chosen == D0
        @warn "No suitable Δ value was found, using the latest found..."
        D_chosen = D
    end

    boh = plot(DD.*30, yaxis=("cose",(-1.0,2.7)), label="Δ*30")
    plot!(boh, (H[1:10:end].-H[1].-0.42)./33, label="E-E[1]", linewidth=0.5)
    plot!(C_H_tot, linewidth=1.5, label="acf")
    plot!(boh, 1:k_max:(maxsteps÷wnd*k_max), τ./2000, label="τ/2e3")
    hline!(boh, [D_chosen*30], label="Δfin")
    gui()

    return X, D_chosen
end


## -------------------------------------
## Evolution
##

HO(x::Float64,ω::Float64) = ω^2*x^2 /2  # not used
LJ(dr::Float64) = 4*(dr^-12 - dr^-6)
der_LJ(dr::Float64) = 4*(6*dr^-8 - 12*dr^-14)   # (dV/dr)/r

function shiftSystem!(A::Array{Float64,1}, L::Float64)
    @inbounds for j = 1:length(A)
        A[j] = A[j] - L*round(A[j]/L)
    end
end


## -------------------------------------
## Thermodinamic Properties
##

function energy(r::Array{Float64,1},L::Float64)
    V = 0.0
    @inbounds for l=0:Int(length(r)/3)-1
        for i=0:l-1
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
(mean(A.*A) - mean(A)^2) * (1-sum((1 .- (1:length(ch))./length(ch)).*ch)*2/length(A))

cv(H::Array{Float64}, T::Float64, ch::Array{Float64}) = variance2(H,ch)/T^2 + 1.5T


function acf(H::Array{Float64,1}, k_max::Int64)

    Z = H .- mean(H)
    C_H = zeros(k_max)

    if k_max>20000
        bar = Progress(k_max, dt=1.0, desc="Calculating acf...", barglyphs=BarGlyphs("[=> ]"), barlen=45)
    end

    @fastmath for k = 1:k_max
        C_H_temp = 0.0
        for i = 1:length(H)-k_max-1
            @inbounds C_H_temp += Z[i] * Z[i+k-1]
        end
        C_H[k] = C_H_temp
        (k_max>20000) && next!(bar)
    end
    CH1 = C_H[1]/(length(H))    # togliere o non togliere k_max, questo è il dilemma

    return C_H ./ (CH1*length(H))    # unbiased and normalized autocorrelation function
end

function fft_acf(H::Array{Float64,1}, k_max::Int)

    Z = H .- mean(H)
    N = length(Z)
    C_H = zeros(k_max)
    fvi = rfft(Z)
    acf = fvi .* conj.(fvi)
    acf = ifft(acf)
    acf = real.(acf)
    C_H = acf[1:k_max]

    return C_H./C_H[1]
end


@fastmath function orderParameter(r::Array{Float64}, L::Float64)
    N = Int(size(r,1)/3)
    Na = round(Int,∛(N/4)) # number of cells per dimension
    a = L / Na  # passo reticolare
    dx = zeros(Na^3*3)
    dy = zeros(Na^3*3)
    dz = zeros(Na^3*3)
    r0, = initializeSystem(N,L)
    dx0 = zeros(Na^3*3)
    dy0 = zeros(Na^3*3)
    dz0 = zeros(Na^3*3)

    @inbounds for k=0:Na^3-1
        for i=1:3
            dx[3k+i] = r[12k+1] - r[12k+3i+1]
            dx[3k+i] -= L*round(dx[3k+i]/L)
            dy[3k+i] = r[12k+2] - r[12k+3i+2]
            dy[3k+i] -= L*round(dy[3k+i]/L)
            dz[3k+i] = r[12k+3] - r[12k+3i+3]
            dz[3k+i] -= L*round(dz[3k+i]/L)
            dx0[3k+i] = r0[12k+1] - r0[12k+3i+1]
            dx0[3k+i] -= L*round(dx0[3k+i]/L)
            dy0[3k+i] = r0[12k+2] - r0[12k+3i+2]
            dy0[3k+i] -= L*round(dy0[3k+i]/L)
            dz0[3k+i] = r0[12k+3] - r0[12k+3i+3]
            dz0[3k+i] -= L*round(dz0[3k+i]/L)
        end
    end
    dr = sqrt.(dx.^2 + dy.^2 + dz.^2)
    R = sqrt.(dx0.^2 + dy0.^2 + dz0.^2)
    K = 2π ./ R
    ordPar = mean(cos.(K.*dr))
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


function make2DtemporalPlot(M::Array{Float64,2}; T=-1.0, rho=-1.0, save=true)
    Plots.default(size=(800,600))
    N = Int(size(M,1)/3)
    L = cbrt(N/rho)
    Na = round(Int,∛(N/4)) # number of cells per dimension
    a = L / Na  # passo reticolare
    X = M[1:3:3N,1]
    #pick the particles near the plane x=a/4
    I = findall(abs.(X -a/4) .< a/4)
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

function simpleReweight(T0::Float64, T1::Float64, O::Array{Float64}, E::Array{Float64})
    if length(O) == length(E)
        return sum(O.*exp.((1/T0-1/T1).*E)) / sum(exp.((1/T0-1/T1).*E))
    else
        @warn "Dimension mismatch between O and E, returning 0..."
        return 0.0
    end
end

function simpleReweight(T0::Float64, TT::Array{Float64}, O::Array{Float64}, E::Array{Float64})
    if length(O[:,1]) == length(E[:,1])
        O2 = zeros(length(TT))
        i = 1
        for T1 in TT
            O2[i] = sum(O.*exp.((1/T0-1/T1).*E)) / sum(exp.((1/T0-1/T1).*E))
            i += 1
        end
        return O2
    else
        @warn "Dimension mismatch between O and E, returning 0..."
        return zeros(length(TT))
    end
end

# fa binning delle energie, crea dei pesi usando boltzmann, che poi normalizza imponendo la media giusta
# poi pesca da E il numero di pesi giusto da ogni bin (attualmente tutti fino a raggiungere numero pesi)
function energyReweight(T0::Float64, T1::Float64, E::Array{Float64})

    ## Binning  (da ricontrollare e velocizzare)
    bins = LinRange(minimum(E), maximum(E), length(E)÷10000)
    δE = bins[2] - bins[1];
    ebin = zeros(Int64, length(E)) # assegna un bin ad ogni E[i]
    nbin = zeros(Int64, length(bins))

    for i=1:length(E)
        @inbounds for j=1:length(bins)
            if E[i] <= bins[j]
                ebin[i] = j
                nbin[j] += 1
                break
            end
        end
    end

    ## Creazione bin ripesati per nuova temperatura
    nbin_new = nbin .* exp.((1/T0-1/T1) .* (bins .+ δE/2)) ./ sum(exp.((1/T0-1/T1) .* (bins .+ δE/2)))
    pesi = nbin_new ./ maximum(nbin_new)

    num, den = 0.0, 0.0
    @inbounds for i=1:length(E) # calcola E media ripesata
        num += pesi[ebin[i]]*E[i]*exp((1/T0-1/T1)*E[i])
        den += pesi[ebin[i]]*exp((1/T0-1/T1)*E[i])
    end
    E2mean = num/den  # energia media ripesata, da forzare
    wrongmean = sum(pesi.*(bins .+ δE/2)) / length(E)
    oldmean = sum(nbin.*(bins .+ δE/2)) / length(E)   # inutile, ma da capire perché è sottostimata
    @show scale_factor = E2mean/wrongmean
    pesi = pesi.*scale_factor
    # ora dev'essere uguale a E2mean, e rappresenta la distribuzione di boltzmann a T1
    #@show sum(pesi.*(bins .+ δE/2)) / length(E)

    ## rescaling (nuovamente) di pesi per far contenere la distribuzione interamente in nbin
    ratio = zeros(Float64, length(pesi))
    @inbounds for j=1:length(nbin)
        # 100 numero arbitrario, se nuovi bin superano vecchi solo in code poco popolate chissene
        if pesi[j]/nbin[j] < 1 || nbin[j] >= 42
            ratio[j] = pesi[j] / nbin[j]
        else
            ratio[j] = 1.0
        end
    end
    # se ratio[qualsiasi] maggiore di 1 vuol dire che la nuova distr si sta prendendo
    # più configurazioni di quelle disponibili, e si riscala pesi di conseguenza
    @show maxratio = maximum(ratio)
    pesi = pesi ./ maxratio
    #plot(nbin, label="orig"); plot!(pesi); gui()

    ## pesca di pesi configurazioni da ogni bin di nbin
    E2 = zeros(length(E))
    count = zeros(Int64, length(pesi))
    for i=1:length(E)   # andrebbero pescati a intervalli (regolari o casuali), non tutti in fila
        if count[ebin[i]] <= pesi[ebin[i]]
            E2[i] = E[i]
            count[ebin[i]] += 1
        end
    end
    filter!(x->x≠0, E2)
    (length(E2)<length(E)/10) && (@warn "Reweighted energy samples low")
    return E2
end


function prettyPrint(T::Float64, rho::Float64, E::Array, P::Array, ch::Array, cv, cv2, OP::Array)
    println("\n")
    println("Mean energy: ", mean(E), " ± ", sqrt(variance2(E,ch)))
    println("Pressure: ", mean(P), " ± ", std(P))
    println("Specific heat: ", cv)
    println("Specific heat (alternative) : ", cv2)
    println("Order parameter : ", mean(OP), " ± ", std(OP))
    println("Average autocorrelation time: ", sum(ch))
    println()
end

function saveCSV(rho, N, T, EE, PP, CV, CV2, C_H, OP)
    data = DataFrame(E=EE, P=PP, CV=CV, CV2=CV2, OP=OP, Ch=[C_H; missings(length(EE)-length(C_H))])
    file = string("./Data/MCtemp_",N,"_rho",rho,"_T",T,".csv")
    CSV.write(file, data)
    @info string("Data saved in ", file)
end

function avg3D(A::Array{Float64,1})
    N = Int(length(A)/3)
    return [sum(A[1:3:N-2]), sum(A[2:3:N-1]), sum(A[3:3:N])]./N
end


# creates an array with length N of gaussian distributed numbers using Box-Muller
function vecboxMuller(sigma::Float64, N::Int, x0=0.0)
    #srand(60)   # sets the rng seed, to obtain reproducible numbers
    x1 = rand(Int(N/2))
    x2 = rand(Int(N/2))
    @. [sqrt(-2sigma*log(1-x1))*cos(2π*x2); sqrt(-2sigma*log(1-x2))*sin(2π*x1)]
end


end
