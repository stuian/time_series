function [MuInf] = MutualInfo(R,F,method)
% ����Ϣ����
%R/F�������Ƕ�ֵͼ��Ҳ�����ǻҶ�ͼ��
%method������Ϣ���һ������Ϣѡ�����ù�һ������Ϣ
if size(F,3) == 3
    F = rgb2gray(F);
end
if size(R,3) == 3
    R = rgb2gray(R);
end;
% ���㻥��Ϣ
Hist = JoinHist(R,F);
[rows,columns] = size(R);
N_Hist = Hist./(rows*columns);%����ֱ��ͼ��һ��
Marg_A = sum(N_Hist); %�������е������
Marg_B = sum(N_Hist,2); %�������е������
H_A = 0;
H_B = 0;
for i=1:1:size(N_Hist,1)   %������H_A
    if Marg_A(i) ~= 0
        H_A = H_A+(-Marg_A(i)*log2(Marg_A(i)));
    end
end
for i=1:1:size(N_Hist,2)   %������H_B
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
     MuInf = 1/MuInf; % �����ǼӺ��ڷ��ӻ����ڷ�ĸ�أ�
 end;
end

%% ͳ��F��R����ͼ�������ֱ��ͼ
function Hist = JoinHist(R,F)
[rows,columns] = size(R);
Hist = zeros(256,256);
for i = 1:1:rows
    for j = 1:1:columns
        Hist(R(i,j)+1,F(i,j)+1) = Hist(R(i,j)+1,F(i,j)+1)+1;
    end
end
end