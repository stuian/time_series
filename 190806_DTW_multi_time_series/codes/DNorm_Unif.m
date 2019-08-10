function DZ = DNorm_Unif(X, card)

% Input: X ts is a subsequence, a row vector
% Output: DZ is a discrete subsequence in range [1,card]

if (~exist('card','var')), card=16; end;

mn = min(X);
mx = max(X);
DZ = round((X-mn)/(mx-mn)*(card-1)) + 1;
DZ=DZ-card/2;
%DZ = zscore(DZ);
end