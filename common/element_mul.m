function FA = element_mul(data3D, Res, mode)
% Superpixel-guided Spectral Attention (Res-based version)
%
% 输入：
%   data3D : H × W × D
%   Res    : seg_im_class_forS3 输出结构
%   mode   : 'minmax' or 'ratio'
%
% 输出：
%   FA     : 加权后的数据 H × W × D

[H, W, D] = size(data3D);
N = H * W;
Num = size(Res.Y, 2);   % 超像素数量

X = reshape(data3D, [], D);

% --- Step 1: 超像素均值 ---
mu = zeros(Num, D);

for i = 1:Num
    Xi = Res.Y{1, i};   % (ni × D)
    
    if isempty(Xi)
        continue;
    end
    
    mu(i, :) = mean(Xi, 1);
end

% --- Step 2: 构建权重 ---
switch mode
    case 'minmax'
        mu_min = min(mu, [], 1);
        mu_max = max(mu, [], 1);
        A_sp = (mu - mu_min) ./ (mu_max - mu_min + eps);
        
    case 'ratio'
        mu_mean = mean(mu, 1);
        A_sp = mu ./ (mu_mean + eps);
        
        % 稳定压缩（关键）
        A_sp = tanh(A_sp);
        
    otherwise
        A_sp = mu ;
end

% --- Step 3: 映射到像素 ---
A_pixel = zeros(N, D);

for i = 1:Num
    idx = Res.index{1, i};   % 当前超像素的像素索引
    A_pixel(idx, :) = repmat(A_sp(i, :), length(idx), 1);
end

% --- Step 4: 点乘 ---
FA = X .* A_pixel;

FA = reshape(FA, H, W, D);

end