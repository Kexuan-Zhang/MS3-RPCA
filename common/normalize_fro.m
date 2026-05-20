function Xn = normalize_fro(X)
    tmp = reshape(X, [], size(X,3));
    tmp = tmp / norm(tmp, 'fro');
    Xn = reshape(tmp, size(X));
end

