function [X_temp] = findConstruct_spectral_spatial(X, index, k, alpha)
% X     : 超像素内像素的光谱特征 (n × m)
% index : 超像素内像素的空间坐标 (n × 2)  [row, col]
% k     : 邻居数量
% alpha : 光谱-空间权重系数 (0~1)，alpha 越大越偏光谱

    if nargin < 4
        alpha = 0.5;   % 默认光谱/空间等权
    end

    if k >= size(X,1)
        k = size(X,1) - 1;
    end

    [n, m] = size(X);
    X_temp = zeros(n, m);

    for i = 1:n
        % ========= 1. 计算光谱距离 =========
        d_spec = EuDist2(X(i,:), X);        % 1 × n

        % ========= 2. 计算空间距离 =========
        d_spat = EuDist2(index(i,:), index);  % 1 × n

        % ========= 3. 距离归一化 =========
        d_spec = d_spec / (mean(d_spec) + eps);
        d_spat = d_spat / (mean(d_spat) + eps);

        % ========= 4. 联合距离 =========
        d_joint = alpha * d_spec + (1 - alpha) * d_spat;

        % ========= 5. 选取 k 个最近邻 =========
        [~, ids] = sort(d_joint);
        neigh_ids = ids(2:k+1);

        temp_x = X(neigh_ids, :);
        temp_d = d_joint(neigh_ids);

        % ========= 6. 特殊情况处理 =========
        if sum(temp_d == 0) >= k - 1
            X_temp(i, :) = X(i, :);
        else
            % ========= 7. 高斯加权 =========
            sigma = mean(temp_d);
            temp_w = exp(-temp_d.^2 / (2 * sigma^2));
            temp_w = temp_w / sum(temp_w);

            X_temp(i, :) = temp_w * temp_x;
        end
    end
end
