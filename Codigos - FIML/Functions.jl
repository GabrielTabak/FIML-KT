module Functions

using LinearAlgebra, Random, Distributions, Statistics
using Symbolics
using MultivariatePolynomials
using DynamicPolynomials
using HomotopyContinuation
const MK = HomotopyContinuation.ModelKit
const ZERO = MK.Expression(0)
const ONE  = MK.Expression(1)
const NEG1 = MK.Expression(-1)

iszeroish(x) = (x == 0)
isoneish(x)  = (x == 1)

export blkdiag, build_blocks, residuals, twoSLS, threeSLS, _minor, det_mk, adj_mk, VpV, build_sets, fill_by_columns, generate_Gamma,
build_R, final_sol, hausman_iter_sem, twoSLS_per_eq, build_blocks_2, hausman_iter_sem_2, generate_B, Method_1,
simulate_sem, build_R_sm, Method_1_1, simulate_sem_given, Method_2, Method_2_1


# Aditional functions
function blkdiag(mats::AbstractMatrix...)
    r = sum(size(m,1) for m in mats)
    c = sum(size(m,2) for m in mats)
    M = zeros(eltype(mats[1]), r, c)
    ri = 1; ci = 1
    for A in mats
        rr, cc = size(A)
        M[ri:ri+rr-1, ci:ci+cc-1] .= A
        ri += rr; ci += cc
    end
    return M
end


function build_blocks(Y,Z,endog_sets, exog_sets, inst_sets)
    T, M = size(Y)
    Xblocks = Vector{Matrix{Float64}}(undef, M)
    Zblocks = Vector{Matrix{Float64}}(undef, M)
    yblocks = Vector{Vector{Float64}}(undef, M)
    for i in 1:M
        Xi = hcat(Y[:, endog_sets[i]], Z[:, exog_sets[i]])
        Zi = Z[:, inst_sets[i]]
        Xblocks[i] = Array(Xi)
        Zblocks[i] = Array(Zi)
        yblocks[i] = Array(Y[:, i])
    end
    return Xblocks, Zblocks, yblocks
end

function residuals(yb, Xb, β::Vector{Float64}, T, M)
    res = zeros(T, M)
    idx = 1
    for i in 1:M
        pi = size(Xb[i],2)
        b_i = β[idx:idx+pi-1]
        res[:, i] = yb[i] .- Xb[i]*b_i
        idx += pi
    end
    return res
end

# projeção por 2SLS por equação (inicial consistente)
function twoSLS_per_eq(yb, Xb, Zb, T, M)
    βhat = Vector{Vector{Float64}}(undef, M)
    for i in 1:M
        Zi, Xi, yi = Zb[i], Xb[i], yb[i]
        Pz = Zi * inv(Zi'Zi) * Zi'
        βhat[i] = inv(Xi'Pz*Xi) * (Xi'Pz*yi)
    end
    return βhat
end



function build_blocks_2(Y,Z,endog_sets, exog_sets, β, B, Gamma)
    
    T, M = size(Y)
    G_it, B_it = fill_by_columns(B, Gamma, β)
    PIinv = B_it*inv(G_it)
    
    Zblocks = Vector{Matrix{Float64}}(undef, M)
    for i in 1:M
        Xi = hcat((Z*PIinv)[:,endog_sets[i]] , Z[:, exog_sets[i]])
        Zblocks[i] = Array(Xi)
    end
    return Zblocks
end



# Estimator 2SLS
function twoSLS(Y::AbstractMatrix, X::AbstractMatrix, B, Gamma;
        endog_sets::Vector{Vector{Int}},
        exog_sets::Vector{Vector{Int}},
        inst_sets::Union{Nothing,Vector{Vector{Int}}}=nothing)
    
    T, M = size(Y)
    K = size(X,2)
    @assert length(endog_sets)==M && length(exog_sets)==M
    
    # instrumentos por equação
    if inst_sets === nothing
        inst_sets = [collect(1:K) for _ in 1:M]   # todos Z como instrumentos
    end

    # constrói blocos de regressoras (X_i) e instrumentos (Z_i) por equação    
    Xb, Zb, yb = build_blocks(Y,X,endog_sets, exog_sets, inst_sets)

    
    βhat = Vector{Vector{Float64}}(undef, M)
    for i in 1:M
        Zi, Xi, yi = Zb[i], Xb[i], yb[i]
        Pz = Zi * inv(Zi'Zi) * Zi'
        βhat[i] = inv(Xi'Pz*Xi) * (Xi'Pz*yi)
    end

    β = vcat(βhat...)    


    G_test, B_test = fill_by_columns(B, Gamma, β)

    return(G_test, B_test)    
#    return β
    
end



# Estimator 3SLS

function threeSLS(Y::AbstractMatrix, X::AbstractMatrix, B, Gamma;
        endog_sets::Vector{Vector{Int}},
        exog_sets::Vector{Vector{Int}},
        inst_sets::Union{Nothing,Vector{Vector{Int}}}=nothing)
    
    T, M = size(Y)
    K = size(X,2)
    @assert length(endog_sets)==M && length(exog_sets)==M
    
    # instrumentos por equação
    if inst_sets === nothing
        inst_sets = [collect(1:K) for _ in 1:M]   # todos Z como instrumentos
    end

    # constrói blocos de regressoras (X_i) e instrumentos (Z_i) por equação    
    Xb, Zb, yb = build_blocks(Y,X,endog_sets, exog_sets, inst_sets)

    Xdiag = blkdiag(Xb...)
    Zsys = blkdiag(Zb...)
    y = vcat(yb...)

    βblocks = twoSLS_per_eq(yb, Xb, Zb, T, M)
    β = vcat(βblocks...)    
    
    U = residuals(yb, Xb, β, T, M)
    Sigma = U'U/T
    Pz = X*inv(X'*X)*X'
    
    β = inv(Xdiag'*kron(inv(Sigma), Pz)*Xdiag)*(Xdiag'*kron(inv(Sigma), Pz)*y)
    
    G_test, B_test = fill_by_columns(B, Gamma, β)

    return(G_test, B_test)   
#    return β
    
end


# Functions to deal with inversion


# submatriz que remove a linha r e a coluna c
function _minor(A::AbstractMatrix{MK.Expression}, r::Int, c::Int)
    n, m = size(A); @assert n == m "Matriz deve ser quadrada"
    M = Matrix{MK.Expression}(undef, n-1, n-1)
    ii = 0
    for i in 1:n
        i == r && continue
        ii += 1
        jj = 0
        for j in 1:n
            j == c && continue
            jj += 1
            M[ii, jj] = A[i, j]
        end
    end
    return M
end

# determinante por expansão de Laplace (primeira linha), com expand simples
function det_mk(A::AbstractMatrix{MK.Expression})::MK.Expression
    n, m = size(A); @assert n == m "Matriz deve ser quadrada"
    if n == 1
        return A[1,1]
    elseif n == 2
        return MK.expand(A[1,1]*A[2,2] - A[1,2]*A[2,1])
    else
        acc = ZERO
        for j in 1:n
            sign = ((1 + j) % 2 == 0) ? ONE : NEG1
            acc  = acc + MK.expand(sign * A[1,j] * det_mk(_minor(A, 1, j)))
        end
        return acc
    end
end

# adjunta(A) = C(A)' onde C_ij = (-1)^{i+j} det(minor(i,j))
function adj_mk(A::AbstractMatrix{MK.Expression})
    n, m = size(A); @assert n == m "Matriz deve ser quadrada"
    C = Matrix{MK.Expression}(undef, n, n)
    for i in 1:n, j in 1:n
        sign = ((i + j) % 2 == 0) ? ONE : NEG1
        C[i,j] = MK.expand(sign * det_mk(_minor(A, i, j)))
    end
    return C'  # adjunta
end


# VpV
function VpV(Y, X, Γ, B)
    V = Y - X * B * inv(Γ)
    
    try 
        return det(V' * V)
    catch _
        return NaN
    end
end


# Build end, ex sets
function build_sets(Gamma_est::AbstractMatrix, Beta_est::AbstractMatrix)
    g = size(Gamma_est, 1)         # nº de endógenas (eqs)
    @assert size(Gamma_est,2) == g "Gamma_est deve ser g×g"
    @assert size(Beta_est,2)  == g "Beta_est deve ter g colunas"

    # Endógenos por equação (coluna de Gamma)
    endog_sets = Vector{Vector{Int}}(undef, g)
    for j in 1:g
        inds = Int[]
        for i in 1:g
            if i != j && !(iszeroish(Gamma_est[i,j]) || isoneish(Gamma_est[i,j]))
                push!(inds, i)
            end
        end
        endog_sets[j] = inds
    end

    # Exógenos por equação (coluna de Beta)
    k = size(Beta_est, 1)
    exog_sets = Vector{Vector{Int}}(undef, g)
    for j in 1:g
        inds = Int[]
        for i in 1:k
            if !iszeroish(Beta_est[i,j])
                push!(inds, i)
            end
        end
        exog_sets[j] = inds
    end

    return endog_sets, exog_sets
end


# Preencher o que recuperou - Metodo iterativo
function fill_by_columns(B::AbstractMatrix, Gamma::AbstractMatrix, aux::AbstractVector)
    k, gB = size(B)
    g1, g2 = size(Gamma)
    @assert gB == g1 == g2 "Dimensões incompatíveis: B é k×g e Gamma deve ser g×g com o mesmo g."

    T = promote_type(eltype(B), eltype(Gamma), eltype(aux))
    G_test = Matrix{T}(I, g1, g2)   # começa como identidade
    B_test = zeros(T, k, gB)        # começa como zeros

    # Contas para validar tamanho de aux
    nfreeG = count(x -> x != 0 && x != 1, Gamma)
    nfreeB = count(!iszero, B)
    needed = nfreeG + nfreeB
    if length(aux) < needed
        throw(ArgumentError("aux tem comprimento $(length(aux)) mas são necessários $needed valores."))
    end

    it = 1
    for j in 1:g1
        # 1) Preenche coluna j de Gamma (apenas entradas livres: ≠ 0 e ≠ 1)
        for i in 1:g1
            γ = Gamma[i, j]
            if γ != 0 && γ != 1
                G_test[i, j] = -aux[it]
                it += 1
            end
        end
        # 2) Preenche coluna j de B (apenas entradas ≠ 0)
        for i in 1:k
            if B[i, j] != 0
                B_test[i, j] = aux[it]
                it += 1
            end
        end
    end

    return G_test, B_test
end



# Generate Gamma, according to identifiability restrictions
function generate_B(k::Int, g::Int, r1::Int, r2::Int, rng=rng)
#    rng = MersenneTwister(123)
    M = zeros(Float64, k, g)
    
    # Passo 1: pelo menos um não-zero em cada linha
    for i in 1:k
        j = rand(rng, 1:g)
        val = -5 + 10 * rand(rng)     
        M[i, j] = val
    end

    # Passo 2: máximo de não-zeros permitido
    max_nonzeros = k*g - r1
    current_nonzeros = count(!iszero, M)
    remaining = max_nonzeros - current_nonzeros

    # Passo 3: preencher o resto respeitando restrição de r2
    positions = [(i,j) for i in 1:k for j in 1:g if M[i,j] == 0]
    shuffle!(rng, positions)

    for (i,j) in positions
        if remaining == 0
            break
        end
        # Verifica se essa coluna já atingiu o limite de não-zeros
        if count(!iszero, M[:,j]) < (k - r2)
            val = -5 + 10 * rand(rng)
            M[i,j] = val
            remaining -= 1
        end
    end
    
    return M
end


function generate_Gamma(g::Int, r::Int, rng=rng)

    M = zeros(Float64, g, g)

    # Passo 1: fixa a diagonal em 1
    for i in 1:g
        M[i,i] = 1.0
    end

    # Passo 2: máximo de não-zeros permitido
    max_nonzeros = g^2 - r
    current_nonzeros = count(!iszero, M)
    remaining = max_nonzeros - current_nonzeros

    # Passo 3: preencher aleatoriamente fora da diagonal
    positions = [(i,j) for i in 1:g for j in 1:g if i != j && M[i,j] == 0]
    shuffle!(rng, positions)

    for (i,j) in Iterators.take(positions, remaining)
#        val = 0
#        while val == 0 || val == 1
#            val = rand(rng, -10:10)
#        end
        val = -5 + 10 * rand(rng)
        M[i,j] = val
    end
    
    return M
end



# Build R matrix use it to ensure which conditions must be zero
function build_R(B::AbstractMatrix, Gamma::AbstractMatrix, g::Integer, k::Integer)
    R_1 = zeros(k*g,k*g)
    which_ones = findall(x -> x == 0, vec(B))

    for i in which_ones
        R_1[i,i] = 1
    end

    R_2 = zeros(g*g,g*g)
    which_ones = findall(x -> x == 0 || x == 1, vec(Gamma))

    for i in which_ones
        R_2[i,i] = 1
    end

    return R_1, R_2
end

# Build R matrix use it to ensure which conditions must be zero 2
function build_R_sm(B::AbstractMatrix, Gamma::AbstractMatrix, g::Integer, k::Integer)
    R_1 = ones(g,g)
    which_ones = findall(x -> x == 0 || x == 1, Gamma)

    for i in which_ones
        R_1[i] = 0
    end

    R_2 = ones(k,g)
    which_ones = findall(x -> x == 0, B)
    for i in which_ones
        R_2[i] = 0
    end

    return R_1, R_2
end



# Retrieve the values in matrix format - Homotopy format
function final_sol(Y, X, solutions_result, B, Gamma, k::Int, g::Int; digits=4)
    best_val = Inf
    best_B   = Matrix{Float64}(undef, k, g)
    best_G   = Matrix{Float64}(undef, g, g)

    for sol in solutions(solutions_result)
        aux = real.(round.(sol; digits=digits))

        B_test = zeros(Float64, k, g)
        G_test = Matrix{Float64}(I, g, g)

        it = 1
        # Preenche G_test nas posições livres de Gamma (nem 0 nem 1)
        for i in 1:g, j in 1:g
            if Gamma[i, j] != 0 && Gamma[i, j] != 1
                G_test[i, j] = aux[it]   # <- era gam[it]; use aux[it]
                it += 1
            end
        end

        # Preenche B_test nas posições não nulas de B
        for i in 1:k, j in 1:g
            if B[i, j] != 0
                B_test[i, j] = aux[it]
                it += 1
            end
        end
        
        
        val = VpV(Y, X, G_test, B_test)
        if isfinite(val) && val < best_val
            best_val = val
            best_B   = copy(B_test)
            best_G   = copy(G_test)
        end
    end

    return best_val, best_G, best_B
end

function hausman_iter_sem(Y::AbstractMatrix, Z::AbstractMatrix, B, Gamma;
        endog_sets::Vector{Vector{Int}},
        exog_sets::Vector{Vector{Int}},
        inst_sets::Union{Nothing,Vector{Vector{Int}}}=nothing,
        initial_guess=nothing,
        maxiter::Int=200, tol::Float64=1e-8, verbose::Bool=true)

    T, M = size(Y)
    K = size(Z,2)
    @assert length(endog_sets)==M && length(exog_sets)==M

    # instrumentos por equação
    if inst_sets === nothing
        inst_sets = [collect(1:K) for _ in 1:M]   # todos Z como instrumentos
    end

    # constrói blocos de regressoras (X_i) e instrumentos (Z_i) por equação
    function build_blocks()
        Xblocks = Vector{Matrix{Float64}}(undef, M)
        Zblocks = Vector{Matrix{Float64}}(undef, M)
        yblocks = Vector{Vector{Float64}}(undef, M)
        for i in 1:M
            Xi = hcat(Y[:, endog_sets[i]], Z[:, exog_sets[i]])
            Zi = Z[:, inst_sets[i]]
            Xblocks[i] = Array(Xi)
            Zblocks[i] = Array(Zi)
            yblocks[i] = Array(Y[:, i])
        end
        return Xblocks, Zblocks, yblocks
    end

    Xb, Zb, yb = build_blocks()
    # empilha em bloco diagonal
    X = blkdiag(Xb...)
    Zsys = blkdiag(Zb...)
    y = vcat(yb...)

    # projeção por 2SLS por equação (inicial consistente)
    function twoSLS_per_eq(yb, Xb, Zb)
        βhat = Vector{Vector{Float64}}(undef, M)
        for i in 1:M
            Zi, Xi, yi = Zb[i], Xb[i], yb[i]
            Pz = Zi * inv(Zi'Zi) * Zi'
            βhat[i] = inv(Xi'Pz*Xi) * (Xi'Pz*yi)
        end
        return βhat
    end
    
    if initial_guess == nothing
        βblocks = twoSLS_per_eq(yb, Xb, Zb)
        β = vcat(βblocks...)
    else
        β = initial_guess
    end
        
        
    # funções auxiliares
    # residuals por equação, dados β
    function residuals(β::Vector{Float64})
        res = zeros(T, M)
        idx = 1
        for i in 1:M
            pi = size(Xb[i],2)
            b_i = β[idx:idx+pi-1]
            res[:, i] = yb[i] .- Xb[i]*b_i
            idx += pi
        end
        return res
    end

    # log-likelihood concentrada (up to constants): - (T/2) log det(Σ)
    llhist = Float64[]
    function conc_loglik(Σ::Matrix{Float64})
        # ignorando termos constantes e |det(B)| (absorvido pela parametrização);
        # útil só como monitor monotônico aproximado
        return -0.5*T*log(det(Σ))
    end

    converged = false
    for it in 1:maxiter
        # (1) estima Σ a partir dos resíduos estruturais
        U = residuals(β)
        Σ = (U'U)/T

        # (2) peso GLS de sistema: Ω^{-1} = Σ^{-1} ⊗ I_T
        Σinv = inv(Symmetric(Σ))
        Winv = kron(Σinv, I(T))

        # (3) passo FIIV/3SLS (sistema IV ponderado)
        #     β_{new} = (X' W Z (Z' W Z)^{-1} Z' W X)^{-1} X' W Z (Z' W Z)^{-1} Z' W y
        ZWZ = Zsys' * Winv * Zsys
        PzW = Zsys * inv(Symmetric(ZWZ)) * Zsys'   # "projeção" ponderada
        XWPZX = X' * Winv * PzW * X
        XWPZy = X' * Winv * PzW * y
        β_new = inv(Symmetric(XWPZX)) * XWPZy

        # (4) checa convergência
        diff = maximum(abs.(β_new .- β))
        push!(llhist, conc_loglik(Σ))
        if verbose
            @info "iter $it |Δβ|∞ = $(round(diff, sigdigits=4))"
        end
        if diff < tol
            β = β_new
            converged = true
            break
        end
        β = β_new
    end
    
    G_test, B_test = fill_by_columns(B, Gamma, β)

    return(G_test, B_test)
    
#    return (β = β, Σ = (residuals(β)'*residuals(β))/T,
#            ll = llhist, converged = converged)
end


function hausman_iter_sem_2(Y::AbstractMatrix, Z::AbstractMatrix, B, Gamma;
        endog_sets::Vector{Vector{Int}},
        exog_sets::Vector{Vector{Int}},
        inst_sets::Union{Nothing,Vector{Vector{Int}}}=nothing,
        initial_guess=nothing,
        maxiter::Int=200, tol::Float64=1e-8, verbose::Bool=true)

    
    T, M = size(Y)
    K = size(Z,2)
    @assert length(endog_sets)==M && length(exog_sets)==M
    
    # instrumentos por equação
    if inst_sets === nothing
        inst_sets = [collect(1:K) for _ in 1:M]   # todos Z como instrumentos
    end

    # constrói blocos de regressoras (X_i) e instrumentos (Z_i) por equação    
    Xb, Zb, yb = build_blocks(Y,Z,endog_sets, exog_sets, inst_sets)
    # empilha em bloco diagonal
    X = blkdiag(Xb...)
    Zsys = blkdiag(Zb...)
    y = vcat(yb...)
    
    
    if initial_guess == nothing
        βblocks = twoSLS_per_eq(yb, Xb, Zb, T, M)
        β = vcat(βblocks...)
    else
        β = initial_guess
    end
    

    converged = false
    for it in 1:maxiter
        # (1) estima Σ a partir dos resíduos estruturais
        U = residuals(yb, Xb, β, T, M)
        Σ = (U'U)/T

        # (2) peso GLS de sistema: Ω^{-1} = Σ^{-1} ⊗ I_T
        Σinv = inv(Symmetric(Σ))
        Winv = kron(Σinv, I(T))

                
        Xhat = build_blocks_2(Y,Z,endog_sets, exog_sets, β, B, Gamma)
        Xhat = blkdiag(Xhat...)
        # (3) passo FIML - FIVE
        
        
        What = Xhat' *Winv
        β_new = inv(What*X)* (What*y)

        # (4) checa convergência
        diff = maximum(abs.(β_new .- β))
        if verbose
            @info "iter $it |Δβ|∞ = $(round(diff, sigdigits=4))"
        end
        if diff < tol
            β = β_new
            converged = true
            break
        end
        β = β_new
    end
    
    G_test, B_test = fill_by_columns(B, Gamma, β)

    return(G_test, B_test)
    
end


# Method 1 - Following Hausman IV
function Method_1(X, Y, B, Gamma, n, g, k, tipo)
    q1 = length(findall(x -> x != 0 , vec(B)))
    q2 = length(findall(x -> x != 0 && x!= 1, vec(Gamma)))

    @var b[1:q1] gam[1:q2]

    Beta_est  = Matrix{MK.Expression}(undef, k, g)
    fill!(Beta_est, ZERO)

    Gamma_est = Matrix{MK.Expression}(undef, g, g)
    for i in 1:g, j in 1:g
        Gamma_est[i,j] = (i == j) ? ONE : ZERO
    end

    # Preenche Beta_est nas posições não nulas de B com b[⋅]
    it = 1
    for i in 1:k, j in 1:g
        if B[i,j] != 0
            Beta_est[i,j] = b[it]
            it += 1
        end
    end

    # Preenche Gamma_est nas posições "desconhecidas" de Gamma com gam[⋅]
    it = 1
    for i in 1:g, j in 1:g
        if Gamma[i,j] != 0 && Gamma[i,j] != 1
            Gamma_est[i,j] = gam[it]
            it += 1
        end
    end


    U = (Y*Gamma_est - X*Beta_est)
    Sigma = adj_mk(U'*U/n)

    Aux = [X X*Beta_est*adj_mk(Gamma_est)]
    

    eqa1 = vec(Aux'*U*Sigma) 
    expand_vector(x) = MK.expand.(vec(x))

    # (1) construa as equações
    eqf_1 = expand_vector(eqa1)          # ::Vector{Expression}


    F = eqf_1 
    vars = [ gam ; b ; ]
    F_sys = System(F; variables=vars)
    F_sys = MK.optimize(F_sys)

        
    if tipo == 1
        result = solve(F_sys)
    else
        result = solve(F_sys ; start_system = :total_degree)
    end
        
    final_r = final_sol(Y, X, result, B, Gamma, k::Int, g::Int; digits=4)
    
    return final_r
    
end 



# Method 1 - Following Hausman IV - Ensuring only correct = 0
function Method_1_1(X, Y, B, Gamma, n, g, k, tipo)
	# Method 1 - Following Hausman
	q1 = length(findall(x -> x != 0 , vec(B)))
	q2 = length(findall(x -> x != 0 && x!= 1, vec(Gamma)))

	@var b[1:q1] gam[1:q2]

	Beta_est  = Matrix{MK.Expression}(undef, k, g)
	fill!(Beta_est, ZERO)

	Gamma_est = Matrix{MK.Expression}(undef, g, g)
	for i in 1:g, j in 1:g
    	Gamma_est[i,j] = (i == j) ? ONE : ZERO
	end

	# Preenche Beta_est nas posições não nulas de B com b[⋅]
	it = 1
	for i in 1:k, j in 1:g
    	if B[i,j] != 0
        	Beta_est[i,j] = b[it]
        	it += 1
   		end
	end

	# Preenche Gamma_est nas posições "desconhecidas" de Gamma com gam[⋅]
	it = 1
	for i in 1:g, j in 1:g
    	if Gamma[i,j] != 0 && Gamma[i,j] != 1
        	Gamma_est[i,j] = gam[it]
        	it += 1
    	end
	end

	R1, R2 = build_R_sm(B,Gamma,g,k) 
	R = [R1;R2]

	U = (Y*Gamma_est - X*Beta_est)
	Sigma = adj_mk(U'*U/n)

	Aux = [X X*Beta_est*adj_mk(Gamma_est)]


	eqa1 = Aux'*U*Sigma
	eqa1 = vec(R .* eqa1)
	expand_vector(x) = MK.expand.(vec(x))

	# (1) construa as equações
	eqf_1 = expand_vector(eqa1)          # ::Vector{Expression}


	F = eqf_1 
	F = F[ .!iszero.(F) ]
	vars = [ gam ; b ; ]
	F_sys = System(F; variables=vars)
	F_sys = HomotopyContinuation.expand.(MK.optimize(F_sys))
        
    if tipo == 1
        result = solve(F_sys)
    else
        result = solve(F_sys ; start_system = :total_degree)
    end
        
    final_r = final_sol(Y, X, result, B, Gamma, k::Int, g::Int; digits=4)
    
    return final_r
    
end 



# Method 2 - Following Hausman CPOs
function Method_2(X, Y, B, Gamma, n, g, k, tipo)
    q1 = length(findall(x -> x != 0 , vec(B)))
    q2 = length(findall(x -> x != 0 && x!= 1, vec(Gamma)))

    @var b[1:q1] gam[1:q2]

    Beta_est  = Matrix{MK.Expression}(undef, k, g)
    fill!(Beta_est, ZERO)

    Gamma_est = Matrix{MK.Expression}(undef, g, g)
    for i in 1:g, j in 1:g
        Gamma_est[i,j] = (i == j) ? ONE : ZERO
    end

    # Preenche Beta_est nas posições não nulas de B com b[⋅]
    it = 1
    for i in 1:k, j in 1:g
        if B[i,j] != 0
            Beta_est[i,j] = b[it]
            it += 1
        end
    end

    # Preenche Gamma_est nas posições "desconhecidas" de Gamma com gam[⋅]
    it = 1
    for i in 1:g, j in 1:g
        if Gamma[i,j] != 0 && Gamma[i,j] != 1
            Gamma_est[i,j] = gam[it]
            it += 1
        end
    end


    U = (Y*Gamma_est - X*Beta_est)
    Sigma = adj_mk(U'*U/n)
    detS = det_mk(U'U/n)
    detG = det_mk(Gamma_est)

	eqa1 = detS*n*adj_mk(Gamma_est') - detG*Y'*(U)*Sigma
	eqa2 = X'*U*Sigma
    

#    eqa1 = vec(Aux'*U*Sigma) 
    expand_vector(x) = MK.expand.(vec(x))

    # (1) construa as equações
    eqf_1 = expand_vector(eqa1)          # ::Vector{Expression}
    eqf_2 = expand_vector(eqa2)

    F = vec([eqf_1; eqf_2])
    vars = [ gam ; b ; ]
    F_sys = System(F; variables=vars)
    F_sys = MK.optimize(F_sys)

        
    if tipo == 1
        result = solve(F_sys)
    else
        result = solve(F_sys ; start_system = :total_degree)
    end
        
    final_r = final_sol(Y, X, result, B, Gamma, k::Int, g::Int; digits=4)
    
    return final_r
    
end 



# Method 2 - Following Hausman CPOs - Ensuring only correct = 0
function Method_2_1(X, Y, B, Gamma, n, g, k, tipo)
    q1 = length(findall(x -> x != 0 , vec(B)))
    q2 = length(findall(x -> x != 0 && x!= 1, vec(Gamma)))

    @var b[1:q1] gam[1:q2]

    Beta_est  = Matrix{MK.Expression}(undef, k, g)
    fill!(Beta_est, ZERO)

    Gamma_est = Matrix{MK.Expression}(undef, g, g)
    for i in 1:g, j in 1:g
        Gamma_est[i,j] = (i == j) ? ONE : ZERO
    end

    # Preenche Beta_est nas posições não nulas de B com b[⋅]
    it = 1
    for i in 1:k, j in 1:g
        if B[i,j] != 0
            Beta_est[i,j] = b[it]
            it += 1
        end
    end

    # Preenche Gamma_est nas posições "desconhecidas" de Gamma com gam[⋅]
    it = 1
    for i in 1:g, j in 1:g
        if Gamma[i,j] != 0 && Gamma[i,j] != 1
            Gamma_est[i,j] = gam[it]
            it += 1
        end
    end
    
    R1, R2 = build_R_sm(B,Gamma,g,k) 


    U = (Y*Gamma_est - X*Beta_est)
    Sigma = adj_mk(U'*U/n)
    detS = det_mk(U'U/n)
    detG = det_mk(Gamma_est)

	eqa1 = detS*n*adj_mk(Gamma_est') - detG*Y'*(U)*Sigma
	eqa2 = X'*U*Sigma
    
    eqa1 = R1.*eqa1
    eqa2 = R2.*eqa2

    expand_vector(x) = MK.expand.(vec(x))

    # (1) construa as equações
    eqf_1 = expand_vector(eqa1)          # ::Vector{Expression}
    eqf_2 = expand_vector(eqa2)

    F = vec([eqf_1; eqf_2])
    F = F[ .!iszero.(F) ]
    vars = [ gam ; b ; ]
    F_sys = System(F; variables=vars)
    F_sys = MK.optimize(F_sys)

        
    if tipo == 1
        result = solve(F_sys; threading=true)
    else
        result = solve(F_sys ; start_system = :total_degree, threading=true)
    end
        
    final_r = final_sol(Y, X, result, B, Gamma, k::Int, g::Int; digits=4)
    
    return final_r
    
end 





function simulate_sem(n::Integer, k::Integer, g::Integer,
                      r1::Integer, r2::Integer, r3::Integer; rng=Random.GLOBAL_RNG)

    # Exógenas
    X = rand(rng, n, k)

    # Sigma SPD para os choques estruturais
    A = randn(rng, g, g)
    Sigma = A' * A                    # SPD

    # Erros U ~ N(0, Sigma) i.i.d. por linha
    d = MvNormal(zeros(g), Sigma)
    U = rand(rng, d, n)'              # n × g

    # Matrizes estruturais (suas funções)
    Gamma = generate_Gamma(g, r2, rng)
    B     = generate_B(k, g, r1, r3, rng)

    # Gerar Y a partir do modelo: Y*Gamma = X*B + U
    Γinv = inv(Gamma)
    Y = (X*B + U) * Γinv

    return Y, X, Gamma, B, Sigma
end



function simulate_sem_given(n::Int, k::Int, g::Int,
					  r1::Integer, r2::Integer, r3::Integer; 
                      X::Union{AbstractMatrix,Nothing}=nothing,
                      Gamma::Union{AbstractMatrix,Nothing}=nothing,
                      B::Union{AbstractMatrix,Nothing}=nothing,
                      rng=Random.GLOBAL_RNG)

    # X
    X = X === nothing ? rand(rng, n, k) : X

    if B == nothing
		B     = generate_B(k, g, r1, r3, rng)
    end

    if Gamma == nothing
		Gamma = generate_Gamma(g, r2, rng)
    end


    Γinv = inv(Gamma)


   # Sigma SPD para os choques estruturais
    A = randn(rng, g, g)
    Sigma = A' * A                    # SPD

    # Erros U ~ N(0, Sigma) i.i.d. por linha
    d = MvNormal(zeros(g), Sigma)
    U = rand(rng, d, n)'              # n × g

    # --------- Construir Y ---------
    Y = (X * B + U) * Γinv      # n×g

	return Y, X, Gamma, B, Sigma
end


end

