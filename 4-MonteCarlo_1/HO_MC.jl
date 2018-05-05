
# toy model of canonical ensemble of harmonic oscillators
function oscillator(; N=500, D=0.5, T0=3.0, maxsteps=10^4)

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
        ap = exp.((LJ.(X,.5) - LJ.(Y,.5))/T0)
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


function equilibriumOSC(X, D, T0)
    eqstepsmax = 2000
    N = Int(length(X))
    j = zeros(eqstepsmax)
    jm = zeros(eqstepsmax÷50)
    Y = zeros(N)
    for n=1:eqstepsmax
        # Proposta
        Y .= X .+ D.*(rand(N).-0.5)
        # P[Y]/P[X]
        ap = exp.((HO.(X,.5) - HO.(Y,.5))/T0)
        η = rand(N)
        for i = 1:N
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
                return X, j[1:n]./(N)
            end
        end
    end
    warn("It seems equilibrium was not reached")
    return X, j./(3N)
end
