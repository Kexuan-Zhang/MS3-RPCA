function Results = seg_im_class_forS3(Y,Ln)
[M,N,B]=size(Y);
Y_reshape=reshape(Y,M*N,B);
Gt=reshape(Ln,[1,M*N]);     %内容是超像素编号
Class=unique(Gt);
Num=size(Class,2);  %num指的是超像素数量
Results.Y=cell(1,Num);
Results.index=cell(1,Num);
for i=1:Num
    Results.index{1,i}=find(Gt==Class(i));  %i个超像素的在gt中的索引
    [m,n] = find(Ln==Class(i));     
    Results.cor{1,i} = [m,n];       %i个超像素的所有像素位置，m和n都是一个列向量
    Results.Y{1,i} =Y_reshape(find(Gt==Class(i)),:);    %i个超像素的像素内容
end
