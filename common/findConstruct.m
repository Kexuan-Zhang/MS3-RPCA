function [X_temp] = findConstruct(X,index,k)  
%X是一个超像素内所有像素数据，index是一个超像素内所有像素的坐标，k是前k个
    if index==0
        index=X;
    end
    if k>=size(X,1)     %k是相邻像素数量，比超像素内像素数量大的话
        k=size(X,1)-1;
    end
    [n m] = size(X);        %n为超像素内像素个数，m为一个像素通道数
    X_temp = zeros(n,m);
    for i=1:n               %遍历当前超像素内所有像素
        dd=EuDist2(index(i,:),index);       %直接坐标算距离
        [~,ids]=sort(dd);       %ids代表排序后的元素在原数组dd中的原始位置
        dd=EuDist2(X(i,:),X(ids(2:k+1),:)); % 计算当前像素i与其位置上邻近的k个像素之间的数据的欧氏距离
        temp_x = X(ids(2:k+1),:);   %位置上最近的k个像素数据
        temp_dd = dd;       % 将距离存储在temp_dd中，就是两个像素数据相减平方
        %如果k个邻近像素中，除了当前像素外，其他像素与当前像素的距离都为0，说明这些像素与当前像素完全相同
        if (sum(temp_dd == 0) == k - 1)
            % 直接将当前像素的值赋给输出矩阵
            X_temp(i, :) = X(i, :);
        else
            % 计算加权系数temp_w，使用高斯核函数进行加权，距离越近的像素权重越大
            temp_w = exp(-dd.^2 / (2 * mean(dd))^2);    %先算距离均值再平方    
            % 将加权系数归一化，确保权重之和为1
            temp_w = temp_w / sum(temp_w);
            % 对 k 个邻近像素进行加权平均，结果存储在输出矩阵 X_temp 中
            X_temp(i, :) = temp_w * temp_x;
        end
    end
end