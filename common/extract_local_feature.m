function [dataDR,data_local] = extract_local_feature(data3D, prevFeature, num_PC, num_Pixel, k, alpha)

[M, N, B] = size(data3D);

% --- ERS ---
labels = cubseg(data3D, num_Pixel);
Res = seg_im_class_forS3(data3D, labels);

A_local = zeros(M*N, B);

Num = size(Res.Y,2);
for i = 1:Num
    Res.Y{1,i} = findConstruct_spectral_spatial(Res.Y{1,i}, Res.cor{1,i}, k, alpha);
    A_local(Res.index{1,i}, :) = Res.Y{1,i};
end

data_local = reshape(A_local, M, N, B);
data_local = element_mul(data_local,Res,'  ');

% --- Feature cat ---
if isempty(prevFeature)
    data_cat = data_local;
else
    data_cat = cat(3, prevFeature, data_local);
end

% --- SuperPCA ---
dataDR = SuperPCA(data_cat, num_PC, labels);

% --- Frobenius norm ---
dataDR=normalize_fro(dataDR);

end