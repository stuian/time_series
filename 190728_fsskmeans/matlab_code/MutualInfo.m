function [MuInf] = MutualInfo(R,F,method)
% 互信息函数
%R/F：可以是二值图像，也可以是灰度图像
%method：互信息与归一化互信息选项，多采用归一化互信息
if size(F,3) == 3
    F = rgb2gray(F);
end
if size(R,3) == 3
    R = rgb2gray(R);
end;
% 计算互信息
Hist = JoinHist(R,F);
[rows,columns] = size(R);
N_Hist = Hist./(rows*columns);%联合直方图归一化
Marg_A = sum(N_Hist); %对所有列单独求和
Marg_B = sum(N_Hist,2); %对所有行单独求和
H_A = 0;
H_B = 0;
for i=1:1:size(N_Hist,1)   %计算熵H_A
    if Marg_A(i) ~= 0
        H_A = H_A+(-Marg_A(i)*log2(Marg_A(i)));
    end
end
for i=1:1:size(N_Hist,2)   %计算熵H_B
    if Marg_B(i) ~= 0
        H_B = H_B + (-Marg_B(i)*log2(Marg_B(i)));
    end
end
H_AB = sum(sum( -N_Hist.*log2(N_Hist+(N_Hist == 0)) ));
 if strcmp(method,'MI')
     MuInf = H_A+H_B-H_AB;
 end
 if strcmp(method,'NMI')
     MuInf = (H_A+H_B)/H_AB;
     MuInf = 1/MuInf; % 到底是加和在分子还是在分母呢？
 end;
end

%% 统计F、R两张图像的联合直方图
function Hist = JoinHist(R,F)
[rows,columns] = size(R);
Hist = zeros(256,256);
for i = 1:1:rows
    for j = 1:1:columns
        Hist(R(i,j)+1,F(i,j)+1) = Hist(R(i,j)+1,F(i,j)+1)+1;
    end
end
end