%this function is to calculate the number of mismatches. 
% You can consider Int_data as the model, Int_line_position is the instance that you want to exam the mismatches
% the function huffmandict is the key one.

function [Bits]=calculate_bit_cost(Int_data,Int_line_position)
% Input: 1.the integer data,
%        2.the reconstruction line of 1. 
% Output: 
%    total_bits :the bit cost of this time series

Int_dist=Int_data-Int_line_position;                              % calculate the distance between Int data and Int reconstruction line;

uniques=unique(Int_dist);             % unique numbers in Int_dist;
numUnique=histc(Int_dist,uniques);    %  how many times each unique number appears

Prob= numUnique/length(Int_dist);

if uniques == 0
    avg = 1;
    Bits = avg*length(Int_data);
else
    
[dict,avg]=huffmandict(uniques,Prob);
Bits=avg*length(Int_data);
end

end