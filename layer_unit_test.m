

%% test vl_nnelemprod

% forward pass
x1 = [1 2 3];
x2 = [1 1 0];
y1 = vl_nnelemprod(x1, x2);
assert(isequal(y1, x1 .* x2));

% backward pass
[dydx1, dydx2] = vl_nnelemprod(x1, x2, [1 1 1]);
assert(isequal(dydx1, x2));
assert(isequal(dydx2, x1));


%% test vl_nnlossregul

% forward pass
y = vl_nnlossregul([4 -5 1], [], 'L1');
assert(isequal(y, 10));

% backward pass
dzdx = vl_nnlossregul([4 -5 1], 1, 'L1');
assert(isequal(dzdx, [1 -1 1]));


%% test vl_nnmixbasis
a = randn(1,1,4,10, 'single');
B = ones(1,1,4*50,10, 'single');

% forward pass
y_fwd = vl_nnmixbasis(a, B, []);

% check if values match numerically for 1 case
y1 = squeeze(y_fwd(1,1,:,1));
a1 = squeeze(a(1,1,:,1));
B1 = squeeze(B(1,1,:,1)); B1 = reshape(B1, [4 50]);
t1 = (a1')*B1;
assert(isequal(t1', y1));

% backward pass
[gradA, gradB] = vl_nnmixbasis(a, B, ones(size(y_fwd), 'single'));
assert(isequal(size(a), size(gradA)));
assert(isequal(size(B), size(gradB)));


%% test vl_nnelemdiv

% forward pass - vector/vector
x1 = [1 2 3];
x2 = [1 1 0];
thresh = 1e-7;
y1 = vl_nnelemdiv(x1, x2);
assert( mean(abs(y1 - [1 2 0])) <= eps('single') );

% forward pass - tensor/scalar
x1 = ones(13,13,1,10);
x2 = 5*ones(1,1,1,10);
y1 = vl_nnelemdiv(x1, x2);
assert( abs(unique(y1) - 0.2) <= eps('single') );


%% backward pass
x1 = [1 2 3];
x2 = [1 1 0];
[dydx1, dydx2] = vl_nnelemdiv(x1, x2, [1 1 1]);
d1 = 1./(x2+thresh) ;
d2 = -x1./(x2.^2 + thresh);

%% 
% vector/scalar
x1 = [1 2 3];
x2 = 5;
[dydx1, dydx2] = vl_nnelemdiv(x1, x2, [1 1 1]);


%%

