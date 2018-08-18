module MC

## TODO
# scelta D in base a media pesata con τ invece che τ migliore, usando vettore D
# aumentare raggio di autocorrelazione per scelta di D
# recuperare dati da fase di termalizzazione?
# limitare calcoli per pressione, che non è così importante
# aggiungere check equilibrio con convoluzione per smoothing e derivata discreta
# ...oppure come da appunti
# parallelizzare e ottimizzare
# autocorrelazione in modo più furbo (fourier, Wiener–Khinchin?)
# kernel openCL?
# provare a riscrivere in C loop simulazione
# individuare zona di transizione di fase (con cv) con loop su temperature
# implementare reweighting per raffinare picco
# trovare picco per diverse ρ
# grafici
# profit

(VERSION >= v"0.7-") && (using Statistics; using Distributed)
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
function metropolis_ST(; N=108, T=.5, rho=.5, Df=1/70, maxsteps=10^6, bmaxsteps=18*10^5, csv=false, anim=false)

    Y = zeros(3N)   # array of proposals
    j = zeros(Int64, maxsteps)  # array di frazioni accettate
    U = zeros(Float64, maxsteps)    # array of total energy
    fstep = 200
    P2 = zeros(Int(maxsteps/fstep))   # virial pressure
    OP = zeros(Int(maxsteps/fstep))  # order parameter

    L = cbrt(N/rho)
    X, a = initializeSystem(N, L)   # creates FCC crystal
    D = a*Df    # Δ iniziale lo scegliamo come frazione di passo reticolare
    X, D = burnin(X, D, T, L, a, bmaxsteps)  # evolve until at equilibrium, while tuning Δ
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

    @time C_H = acf(H, 35000)
    τ = sum(C_H)
    CV = cv(H,T,C_H)
    CV2 = variance(H[1:ceil(Int,τ/5):end])/T^2 + 1.5T
    prettyPrint(T, rho, H, P, C_H, CV, CV2, OP)
    csv && saveCSV(rho, N, T, H, P, CV, CV2, C_H, OP)
    ##anim && makeVideo(XX, T=T, rho=rho, D=D)

    return H, P, j./(3N), C_H, CV, CV2, OP
end


## WIP: doesn't save arrays (expected) but neither can calculate CV
# multi process implementation, using pmap
function metropolis_MP(; N=108, T=.7, rho=.2, Df=1/70, maxsteps=10^6)

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

# al momento setta solo D, vorremmo che facesse raggiungere anche l'eq termodinamico
# DA RISCRIVERE USANDO 2 LOOP
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

            C_H = acf(H[n-wnd+1:n], k_max)  # autocorrelation function in current window
            C_H_tot = [C_H_tot; C_H]
            τ[n÷wnd] = sum(C_H)
            # if τ results negative, we put it to a high number to simplify the next conditionals
            τ[n÷wnd]<0 ? τ[n÷wnd]=42e5 : nothing

            # average acceptance ratio in current window
            jm[n÷wnd] = mean(j[(n-wnd+1):n])./(3N)
            println("\nAcceptance ratio = ", round(jm[n÷wnd]*1e4)/1e4, ",\t τ = ", round(τ[n÷wnd]*1e4)/1e4)

            if jm[n÷wnd] > 0.25 && jm[n÷wnd] < 0.7
                # if acceptance rate is good, choose D to minimize autocorrelation
                # the first condition excludes the τ values found in the first 3 windows,
                # since equilibrium has not been reached yet.
                if n>wnd*3 && τ[n÷wnd] < minimum(τ[4:n÷wnd-1])
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

    D_chosen == D0 && warn("No suitable Δ value was found, using default...")

    boh = plot(DD.*30, yaxis=("cose",(-1.0,2.7)), label="Δ*30")
    plot!(boh, (H[1:10:end].-H[1].-0.42)./33, label="E-E[1]", linewidth=0.5)
    plot!(C_H_tot, linewidth=1.5, label="acf")
    plot!(boh, 1:k_max:(maxsteps÷wnd*k_max), τ./2000, label="τ/2000")
    hline!(boh, [D_chosen*30], label="final Δ*30")
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
(mean(A.*A) - mean(A)^2) * (1-sum((1 .- (1:length(ch))./length(ch)).*ch)*2/length(A))

cv(H::Array{Float64}, T::Float64, ch::Array{Float64}) = variance2(H,ch)/T^2 + 1.5T


# Da velocizzare
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
    CH1 = C_H[1]/(length(H)-k_max)

    return C_H ./ (CH1*(length(H)-k_max))    # unbiased and normalized autocorrelation function
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
    info("Data saved in ", file)
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
