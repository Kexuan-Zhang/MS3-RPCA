clc;clear;close all
addpath(genpath(cd));

%% ================= Configuration =================
PCA_dims = [30 30 30];
trainnum_list = [3 5 7 9 11 13 15];
iterNum = 10;

database = 'Indian';

%% ================= Load Dataset =================
if strcmp(database,'Indian')
    load Indian_pines_corrected; load Indian_pines_gt; load Indian_pines_randp
    data3D = indian_pines_corrected;  label_gt = indian_pines_gt;
    num_SuperPixels = [80 40 20];
    k =43; alpha =0.2;
elseif strcmp(database,'PaviaU')
    load PaviaU; load PaviaU_gt; load PaviaU_randp
    data3D = paviaU;  label_gt = paviaU_gt;
    num_SuperPixels = [120 60 30];
    k = 45 ; alpha =0.5;
end

data3D = data3D ./ max(data3D(:));
[M,N,B] = size(data3D);

%% ================= Step 1: Multi-Scale Local Feature Extraction =================

[dataDR1, local1] = extract_local_feature( ...
    data3D, [], PCA_dims(1), num_SuperPixels(1), k, alpha);

[dataDR2, local2] = extract_local_feature( ...
    data3D, dataDR1, PCA_dims(2), num_SuperPixels(2), k, alpha);

[feat_local, local3] = extract_local_feature( ...
    data3D, dataDR2, PCA_dims(3), num_SuperPixels(3), k, alpha);


%% ================= Step 2: Multi-Scale Global Feature Extraction =================
data_global = cat(3, local1, local2, local3);

% PCA
[n,m,fL] = size(data_global);
data_global = reshape(data_global, n*m, fL);
[P] = Eigenface_f(data_global', 30);
feat_global = data_global*P;
feat_global = reshape(feat_global, n, m, 30);

% Normalization with Frobenius norm
feat_global = normalize_fro(feat_global);

%% ================= Step 3: Multi-Scale Feature Fuse =================
feat_fused = cat(3, feat_local, feat_global);

% PCA
[n,m,f]=size(feat_fused);
feat_fused = reshape(feat_fused,n*m,f);
[P] = Eigenface_f(feat_fused',30);
feat_final = feat_fused*P;
feat_final = reshape(feat_final, n, m, 30);

%% ================= Step 4: SVM Classification =================

all_results_mean = zeros(1,length(trainnum_list));
all_results_std  = zeros(1,length(trainnum_list));

for tt = 1:length(trainnum_list)
    trainnum = trainnum_list(tt);
    fprintf('\n===== trainnum = %d =====\n', trainnum);

    accy_best = zeros(1,iterNum);
    
    parfor iter = 1:iterNum

        randpp = randp{iter};
        [DataTest,DataTrain,CTest,CTrain,~] = ...
            samplesdivide(feat_final,label_gt,trainnum,randpp);
    
        trainlabel = getlabel(CTrain);
        testlabel  = getlabel(CTest);
    
        GA = [0.01 0.1 1 5 10 15 20 30 40 50 100:100:500];
        accy = zeros(1,length(GA));
    
        for g = 1:length(GA)
            gamma = GA(g);
            cmd = ['-q -c 100000 -g ' num2str(gamma) ' -b 1'];
    
            model = svmtrain(trainlabel', DataTrain, cmd);
            [predict_label, ~, ~] = ...
                svmpredict(testlabel', DataTest, model, '-b 1');
    
            [~, accuracy1] = confusion_matrix(predict_label', CTest);
            accy(g) = accuracy1;
        end
    
        accy_best(iter) = max(accy);   % parfor
    end

    all_results_mean(tt) = mean(accy_best);
    all_results_std(tt)  = std(accy_best);

    fprintf('trainnum = %d → OA = %.4f ± %.4f\n', ...
        trainnum, all_results_mean(tt), all_results_std(tt));
end

%% ================= Report =================
fprintf('\n========== Results ==========\n');
fprintf('trainnum: ');
fprintf('%d ', trainnum_list);
fprintf('\n');

fprintf('Average OA:     \n');
for i = 1:length(trainnum_list)
    fprintf('%.4f ± %.4f\n', all_results_mean(i), all_results_std(i));
end
