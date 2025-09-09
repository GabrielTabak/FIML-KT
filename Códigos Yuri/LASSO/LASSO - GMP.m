
n = 500;   % número de linhas
k = 2;     % número de colunas
d = 7;   a = exp(-1/sqrt(d)); 

% Sorteios aleatórios
u = randn(n, 1);    % vetor n x 1 com distribuição normal(0,1)
X = randn(n, k);

beta = [0.1 ; 2];     Y = X*beta + u; 

% Função N(b)
N = @(b) prod( b + a.^(1:d-1) );
p = @(x) x .* ( N(x) - N(-x) ); q = @(x) N(x) + N(-x);

% resolução

mpol('b', k, 1)   % b é kx1: (b1 , ... , bk)'
mpol('r', k, 1)

Qb = (Y - X*b)'*(Y - X*b)

constr = [];
constr = [constr; sum(r) == 1];        % escalar
constr = [constr; r >= 0];              % k×1
constr = [constr; 2 - (b'*b) >= 0];     % bola L2 
constr = [constr; 2 - (r'*r) >= 0]; 

for i = 1:k
    constr = [constr; p(b(i)) - r(i)*q(b(i)) == 0];
end

P = msdp(min(Qb), constr, 7);

% Resolver
[status,obj] = msol(P);

% Extrair solução
b_sol = double(b)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Sparcity 

