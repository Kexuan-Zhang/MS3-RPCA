function reduced3DMatrix = applyPCAon3DMatrix(inputMatrix)
    % 输入参数：inputMatrix为一个三维矩阵（MxNxP）
    
    % 获取输入矩阵的尺寸
    [M, N, P] = size(inputMatrix);
    
    % 将前两维合并为一维
    reshapedMatrix = reshape(inputMatrix, M * N, P);
    
    % 对新的二维矩阵进行PCA
    [P] = Eigenface_f(reshapedMatrix',30);
    A_PC=reshapedMatrix*P;
    
    % 将结果reshape回三维矩阵，大小为 (M, N, 30)
    reduced3DMatrix = reshape(A_PC, M, N, 30);
end

