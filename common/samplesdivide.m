function [DataTest, DataTrain, CTest, CTrain, indian_pines_map] = samplesdivide(indian_pines_corrected,indian_pines_gt,train,randpp)

[m, n, p] = size(indian_pines_corrected);
CTrain = [];CTest = [];
DataTest  = [];
DataTrain = [];
indian_pines_map = uint8(zeros(m,n));
data_col = reshape(indian_pines_corrected,m*n,p);

for i = 1:max(indian_pines_gt(:))   %遍历每个类别，进行样本划分。这里是获取分类的最大编号
    ci = length(find(indian_pines_gt==i));   %当前类别的样本数量 
    [v]=find(indian_pines_gt==i);    %当前类别的像素索引
    datai = data_col(find(indian_pines_gt==i),:);   %当前类别的光谱数据，形状为 [ci, p]
%     if train>1
%         cTrain = round(train);
%     elseif train<1
%         cTrain = round(ci*train);
%     end
%     if train>ceil(ci/2)
%         cTrain = ceil(ci/2);
%     end
    %将样本数量直接设定，不是百分比
    cTrain=train;        
    if train>ceil(ci/2)
       cTrain = ceil(ci/2);
    end
    cTest  = ci-cTrain;
    CTrain = [CTrain cTrain];   %后面类的数据需要接上去
    CTest = [CTest cTest];
    index = randpp{i};  %获取当前类别的随机索引，index大小刚好对应该类的样本数量
    DataTest = [DataTest; datai(index(1:cTest),:)];     %前 cTest 个索引用于测试集，后面的类数据也要接上去
    DataTrain = [DataTrain; datai(index(cTest+1:cTest+cTrain),:)];    %后 cTrain 个索引用于训练集
   
    indian_pines_map(v(index(1:cTest))) = i;   %将测试集的像素在 indian_pines_map 中标记为当前类别的编号 i 
end

%% Normalize
DataTest = fea_norm(DataTest);
DataTrain = fea_norm(DataTrain);
 