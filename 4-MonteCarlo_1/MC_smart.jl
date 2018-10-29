module MCs

using Statistics, FFTW, Distributed, Random, DataFrames, CSV, ProgressMeter, PyCall, Plots
pyplot()
fnt = "sans-serif"
default(titlefont=Plots.font(fnt,24), guidefont=Plots.font(fnt,24),
tickfont=Plots.font(fnt,14),
legendfont=Plots.font(fnt,14))

# add missing directories in current folder
any(x->x=="Data", readdir("./")) || mkdir("Data")
any(x->x=="Plots", readdir("./")) || mkdir("Plots")
any(x->x=="Video", readdir("./")) || mkdir("Video")

# Main function, it creates the initial system, runs a (long)
# burn-in for thermalization
# and Δ selection and then runs a Monte Carlo simulation for maxsteps
function metropolis_ST(; N=32, T=.5, rho=.5, Δ=2e-3, A=0.00008, γ=0.065, maxsteps=10^7, bmaxs=72*10^4, csv=false, anim=false)

    Y = zeros(3N)   # array of proposals
    j = zeros(Int64, maxsteps)  # array di frazioni accettate
    ap = zeros(N) # acceptance probability
    U = zeros(Float64, maxsteps)    # array of total energy
    fstep = 100
    XX = zeros(3N, Int(maxsteps/fstep))
    FF = zeros(3N, Int(maxsteps/fstep))    #TEMPORANEO
    P2 = zeros(Int(maxsteps/fstep))   # virial pressure
    OP = zeros(Int(maxsteps/fstep))  # order parameter

    L = cbrt(N/rho)
    X, a = initializeSystem(N, L)   # creates FCC crystal
    X, A = burnin(X, A, T, L, a, bmaxs)  # evolve until equilibrium, while tuning Δ
    #A = γ*T
    #σ = sqrt(4A*Δ)/γ

    prog = Progress(maxsteps, dt=1.0, desc="Simulating...", barglyphs=BarGlyphs("[=> ]"), barlen=50)
    @inbounds for n = 1:maxsteps
        if (n-1)%fstep == 0
            i = cld(n,fstep)
            XX[:,i] = X
            FF[:,i] = forces(X,L)
            P2[i] = vpressure(X,L)
        end
        U[n] = energy(X,L)
        X, j[n] = oneParticleMove!(X, Y, N, L, A, T)
        # ap = markovProbability!(X, Y, N, L, A, T)
        next!(prog)
    end
    H = U.+3N*T/2
    P = P2.+rho*T
    @show mean(j)/N
    #@show mean(en)/mean(wn)
    #@show mean(en.-wn)

    C_H = fft_acf(H, 42000)   # not more than ~36k if using non-fft acf
    τ = sum(C_H)
    CV = cv(H,T,C_H)
    CV2 = variance(H[1:ceil(Int,τ/2):end])/T^2 #  + 1.5T
    prettyPrint(T, rho, H, P, C_H, CV, CV2, OP)
    csv && saveCSV(rho, N, T, H, P, CV, CV2, C_H, OP)

    return H, P, j./(N), C_H, CV, CV2, XX, AP, en, wn
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
    X .+= a.*(rand(length(X)).-0.5)./100
    return X, a
end

# bisognerebbe controllare meglio quando si è raggiunto l'eq termodinamico
function burnin(X::Array{Float64}, A0::Float64, T::Float64, L::Float64, a::Float64, maxsteps::Int64)

    wnd = maxsteps ÷ 12 # larghezza finestra su cui fissare D e calcolare tau
    k_max = wnd ÷ 5  # distanza massima per autocorrelazione
    N = Int(length(X)/3)
    j = zeros(maxsteps)
    jm = zeros(maxsteps÷wnd)
    Y = zeros(3N)
    ap = zeros(3N) #acceptance probability
    DD = zeros(maxsteps÷wnd*k_max)    # solo per grafico stupido
    U = zeros(maxsteps)
    C_H_tot = []
    τ = ones(maxsteps÷wnd).*1e6

    A_chosen = A0    # D da restituire, minimizza autocorrelazione
    A = A0
    #γ = γ0
    #A = γ*T
    #σ = sqrt(4A*Δ)/γ

    # pre-thermalization
    @inbounds for n = 1:50000
        #ap = markovProbability!(X, Y, N, L, A, T)
        #η = rand(3N)
        X, j[n] = oneParticleMove!(X, Y, N, L, A, T)
    end
    @show mean(j[1:50000])/(N)
    j = zeros(maxsteps)

    # γ selection + moar thermalization
    for n=1:maxsteps
        U[n] = energy(X,L)
        #ap = markovProbability!(X, Y, N, L, A, T)
        #η = rand(3N)
        X, j[n] = oneParticleMove!(X, Y, N, L, A, T)

        # ogni wnd passi calcola autocorr e aggiorna D in base ad acceptance ratio e τ
        if n%wnd == 0
            DD[Int(n÷wnd*k_max-k_max+1):Int(n÷wnd*k_max)] .= A # per grafico stupido
            # autocorrelation function in current window
            C_H = fft_acf(U[n-wnd+1:n], k_max)
            C_H_tot = [C_H_tot; C_H]
            τ[n÷wnd] = sum(C_H)

            # average acceptance ratio in current window
            jm[n÷wnd] = mean(j[(n-wnd+1):n])/(N)
            println("\nAcceptance ratio = ", round(jm[n÷wnd]*1e4)/1e4, ",\t τ = ",
             round(τ[n÷wnd]*1e4)/1e4, "\t E = ", mean(U[(n-wnd+1):n])+3N*T/2)

            # if acceptance rate is good, choose D to minimize autocorrelation
            # the first condition excludes the τ values found in the first 3 windows,
            # since equilibrium has not been reached yet (probably).
            if jm[n÷wnd] > 0.6 && jm[n÷wnd] < 0.9

                #if n>wnd*3 && abs(τ[n÷wnd]) < minimum(abs.(τ[3:n÷wnd-1]))
                if n>wnd*3 && abs(τ[n÷wnd]) < minimum(abs.(τ[3:n÷wnd-1]))
                    @show A_chosen = A
                end
                @show A = A_chosen*(1 + rand()/2 - 0.25)
                #A = γ*T
                #σ = sqrt(4A*Δ)/γ

            elseif jm[n÷wnd] < 0.6
                @show A = A/1.3
                #A = γ*T
                #σ = sqrt(4A*Δ)/γ
                τ[n÷wnd] = 1e6  # big value, so it won't get chosen
            else
                @show A = A*1.3
                #A = γ*T
                #σ = sqrt(4A*Δ)/γ
                τ[n÷wnd] = 1e6
            end
        end
    end

    if A_chosen == A0
        @warn "No suitable Δ value was found, using the latest found..."
        A_chosen = A
    end

    boh = plot(DD.*2e7, yaxis=("cose",(-1.0,2.7)), label="Δ*2e7", reuse=false)
    plot!(boh, (U[1:5:end].-U[1])./15 .+1.0, label="E-E[1]", linewidth=0.5)
    plot!(C_H_tot, linewidth=1.5, label="acf")
    plot!(boh, 1:k_max:(maxsteps÷wnd*k_max), τ./100, label="τ/100")
    hline!(boh, [A_chosen*2e7], label="Δfin", reuse=false)
    gui()

    return X, A_chosen
end


## -------------------------------------
## Evolution
##

global AP, en, wn = [.0], [.0], [.0]

function markovProbability!(X::Array{Float64,1}, Y::Array{Float64,1}, N::Int, L::Float64, A::Float64, T::Float64)

    Wmn = zeros(N)  # biased weight from the current step to the proposed one
    Wnm = zeros(N)  # biased weight from the proposed step to the current one
    ap = zeros(N)

    gauss = vecboxMuller(sqrt(2A),3N)    # 3N se in cartesiane
    ϕ = rand(N)*2*π
    θ = rand(N)*π
    displ = gauss
    # for n=1:N   # da precalcolare?
    #     displ[3n-2] = gauss[n]*cos(ϕ[n])*sin(θ[n])
    #     displ[3n-1] = gauss[n]*sin(ϕ[n])*sin(θ[n])
    #     displ[3n-0] = gauss[n]*cos(θ[n])
    # end

    FX = forces(X,L)
    #@show mean(FX), std(FX)

    #Y .= X .+ D.*FX .+ gauss.*sigma     #force da dividere per γ?
    deltaX = FX.*(A/T) .+ displ   # usare shiftSystem()?
    #make3Dplot(FX.*(A/T))
    #@show mean(deltaX), std(deltaX)
    Y .= X .+ deltaX
    make3Dplot(Y, rho=0.24)
    gui()
    FY = forces(Y,L)
    #make3Dplot(FY)
    deltaF = FY .- FX
    sumF = FY .+ FX
    shiftSystem!(Y,L)   # probably useless here
    #@show mean(FY), std(FX)

    for n=1:N   #ricontrollare segni, pedici
        #Wmn[n] = (deltaX[3n-2]-FX[3n-2].*(A/T))^2 + (deltaX[3n-1]-FX[3n-1].*(A/T))^2 + (deltaX[3n]-FX[n].*(A/T))^2
        #Wnm[n] = (deltaX[3n-2]+FY[3n-2].*(A/T))^2 + (deltaX[3n-1]+FY[3n-1].*(A/T))^2 + (deltaX[3n]+FY[3n].*(A/T))^2
        #@show Wmn[n] - Wnm[n]
        #println()
        deltaW = (deltaF[3n-2]^2 + deltaF[3n-1]^2 + deltaF[3n]^2
        + 2*(deltaF[3n-2]*FX[3n-2] + deltaF[3n-1]*FX[3n-1] + deltaF[3n]*FX[3n])) * A/(4T)
        ap[n] = exp(-(energy(Y,L)-energy(X,L) + (deltaX[3n-2]*sumF[3n-2] + deltaX[3n-1]*sumF[3n-1] + deltaX[3n]*sumF[3n])/2 + deltaW)/T)
    end
    #WX = (deltaX .- FX).^2  #controllare segni
    #WY = (-1 .* deltaX .- forces(Y,L)).^2
    #println()

    #ap = exp.((energy(X,L) - energy(Y,L))/T .+ (Wnm.-Wmn)./(4*A))

    push!(en, (energy(X,L)-energy(Y,L))/T)
    push!(wn, mean((Wnm.-Wmn)./(4*A)))
    push!(AP, mean(ap))
    #println(" ")
    @show return ap
end

function oneParticleMove!(X::Array{Float64,1}, Y::Array{Float64,1}, N::Int, L::Float64, A::Float64, T::Float64)

    displ = vecboxMuller(sqrt(2A),3N)
    η = rand(N)
    j = 0

    @inbounds for n = randperm(N)
        #println()
        #@show n
        #make3Dplot(X, rho=0.24)
        Um = energySingle(X,L,n-1)
        Fm = force(X,L,n-1)
        #@show mean(Fm), std(Fm)

        deltaX = Fm[1]*(A/T) + displ[3n-2]
        deltaY = Fm[2]*(A/T) + displ[3n-1]
        deltaZ = Fm[3]*(A/T) + displ[3n]
        Y .= X
        Y[3n-2] = X[3n-2] + deltaX
        Y[3n-1] = X[3n-1] + deltaY
        Y[3n-0] = X[3n-0] + deltaZ

        #make3Dplot(Fm, rho=0.24)
        Un = energySingle(Y,L,n-1)
        Fn = force(Y,L,n-1)
        #@show mean(Fn), std(Fn)
        deltaF = Fn .- Fm
        sumF = Fn .+ Fm
        shiftSystem!(Y,L)   # probably useless here

        deltaW = (deltaF[1]^2 + deltaF[2]^2 + deltaF[3]^2
        + 2*(deltaF[1]*Fm[1] + deltaF[2]*Fm[2] + deltaF[3]*Fm[3])) * A/(4T)

        ap = exp(-(Un-Um + (deltaX*sumF[1] + deltaY*sumF[2] + deltaZ*sumF[3])/2 + deltaW)/T)

        ## metodo alternativo
        #Wmn = (deltaX-Fm[1]*(A/T))^2 + (deltaY-Fm[2]*(A/T))^2 + (deltaZ-Fm[3].*(A/T))^2
        #Wnm = (deltaX+Fn[1]*(A/T))^2 + (deltaY+Fn[2]*(A/T))^2 + (deltaZ+Fn[3].*(A/T))^2
        #ap2 = exp((energy(X,L) - energy(Y,L))/T + (Wnm-Wmn)/(4*A))
        #@show ap, ap2

        if η[n] < ap
            X[3n-2] = Y[3n-2]
            X[3n-1] = Y[3n-1]
            X[3n] = Y[3n]
            j += 1
        end

    end
    return X, j
end


LJ(dr::Float64) = 4*(dr^-12 - dr^-6)
der_LJ(dr::Float64) = 4*(6*dr^-8 - 12*dr^-14)   # (dV/dr)/r

function shiftSystem!(A::Array{Float64,1}, L::Float64)
    @inbounds for j = 1:length(A)
        A[j] = A[j] - L*round(A[j]/L)
    end
end

function shiftSystem(A::Array{Float64,1}, L::Float64)
    B = Array{Float64}(undef, length(A))
    @inbounds for j = 1:length(A)
        B[j] = A[j] - L*round(A[j]/L)
    end
    return B
end


## -------------------------------------
## Thermodinamic Properties
##

function energy(r::Array{Float64,1}, L::Float64)
    V = 0.0
    @inbounds for l=1:Int(length(r)/3)-1
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
                V += 1.0/(dr2^3)^2 - 1.0/dr2^3
            end
        end
    end
    return V*4
end

function energySingle(r::Array{Float64,1}, L::Float64, i::Int)
    V = 0.0
    for l=0:Int(length(r)/3)-1
        if i != l
            dx = r[3l+1] - r[3i+1]
            dx = dx - L*round(dx/L)
            dy = r[3l+2] - r[3i+2]
            dy = dy - L*round(dy/L)
            dz = r[3l+3] - r[3i+3]
            dz = dz - L*round(dz/L)
            dr2 = dx*dx + dy*dy + dz*dz
            if dr2 < L*L/4
                #V += LJ(sqrt(dr2))
                V += 1.0/(dr2^3)^2 - 1.0/dr2^3
            end
        end
    end
    return V*4
end



function forces(r::Array{Float64,1}, L::Float64)
    F = zeros(length(r))
    for l=1:Int(length(r)/3)-1
         for i=0:l-1
            dx = r[3l+1] - r[3i+1]
            dx = dx - L*round(dx/L)
            dy = r[3l+2] - r[3i+2]
            dy = dy - L*round(dy/L)
            dz = r[3l+3] - r[3i+3]
            dz = dz - L*round(dz/L)
            dr2 = dx*dx + dy*dy + dz*dz
            (dr2 == .0 ) && @warn "WTF"
            (dr2 < 1e-9) && @warn "Particles too near each other"
            if dr2 < L*L/4
                #dV = -der_LJ(sqrt(dr2)) #add smoothing?
                dV = -24/(dr2*dr2*dr2*dr2) + 48/(dr2^3*dr2^2*dr2^2)
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

function force(r::Array{Float64,1}, L::Float64, i::Int)
    FX, FY, FZ = 0.0, 0.0, 0.0
    for l=0:Int(length(r)/3)-1
        if i != l
            dx = r[3l+1] - r[3i+1]
            dx = dx - L*round(dx/L)
            dy = r[3l+2] - r[3i+2]
            dy = dy - L*round(dy/L)
            dz = r[3l+3] - r[3i+3]
            dz = dz - L*round(dz/L)
            dr2 = dx*dx + dy*dy + dz*dz
            #(dr2 == .0 ) && @warn "WTF"
            #(dr2 < 1e-9) && @warn "Particles too near each other"
            if dr2 < L*L/4
                #dV = -der_LJ(sqrt(dr2)) #add smoothing?
                dV = -24/(dr2*dr2*dr2*dr2) + 48/(dr2^3*dr2^2*dr2^2)
                FX += dV*dx
                FY += dV*dy
                FZ += dV*dz
            end
        end
    end
    return [FX, FY, FZ]
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

cv(H::Array{Float64}, T::Float64, ch::Array{Float64}) = variance2(H,ch)/T^2# + 1.5T


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
    fvi = rfft(Z)
    acf = ifft(fvi .* conj.(fvi))
    acf = real.(acf)

    return acf[1:k_max]./acf[1]
end



## -------------------------------------
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
    I = findall(abs.(X .-a/4) .< a/4)
    # now the X indices of M in the choosen plane are 3I-2
    scatter(M[3I.-1,1], M[3I,1], m=(7,0.7,:red,Plots.stroke(0)),w=7, xaxis=("x",(-L/2,L/2)), yaxis=("y",(-L/2,L/2)), leg=false)
    for i =2:size(M,2)
        scatter!(M[3I.-1,i], M[3I,i], m=(7,0.05,:blue,Plots.stroke(0)), markeralpha=0.05)
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

    ## Binning (da ricontrollare e velocizzare)
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
        if pesi[j]/nbin[j] < 1 || nbin[j] >= 69
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
    println("Mean energy: ", mean(E), " ± ", sqrt(abs(variance2(E,ch))))
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
@inbounds function vecboxMuller(sigma::Float64, N::Int, x0=0.0)
    #srand(60)   # sets the rng seed, to obtain reproducible numbers
    x1 = rand(Int(N/2))
    x2 = rand(Int(N/2))
    @. sigma*[sqrt(-2*log(x1))*cos(2π*x2); sqrt(-2*log(x2))*sin(2π*x1)] + x0
end


end
