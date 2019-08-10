
function Bits=calculate_model_cost(Int_data)


uniques=unique(Int_data);             % unique numbers in Int_dist;
numUnique=histc(Int_data,uniques);    %  how many times each unique number appears

Prob= numUnique/length(Int_data);

if uniques == 0
    avg = 1;
    Bits = avg*length(Int_data);
else
    
    [dict,avg]=huffmandict(uniques,Prob);
    Bits=avg*length(Int_data);
    
end