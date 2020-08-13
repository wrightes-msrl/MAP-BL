%% Introduction to Bayesian Learning
% Suppose a bird population is made up of birds that suffer from heavy metal 
% contamination and others that don’t. Our goal is to estimate the proportion, 
% $p$, of birds that suffer from the contamination. At first, we might know nothing 
% about the bird population at all. In this absence of information, we propose 
% a number of possible hypotheses that describe the value of $p$:
%% 
% * $H_1:\,0\le p<0.2$
% * $H_2:\,0.2\le p<0.4$
% * $H_3:\,0.4\le p<0.6$
% * $H_4:\,0.6\le p<0.8$
% * $H_5:\,0.8\le p\le 1$
%% 
% Without any prior information about our bird population, we have no way of 
% determining which of these hypotheses are correct. Therefore, the best we can 
% do is assume that any one of these hypotheses is just as likely as being correct 
% as any of the others.
% 
% They are all equally likely to be correct:
% 
% $$P(H_1 ) = P(H_2 ) = P(H_3 ) = P(H_4 ) = P(H_5 ) =\frac{1}{5}=0.2$$
% 
% We call these probabilities the prior probabilities for the five competing 
% hypotheses, and as you can see, they are for now all equal. We'll choose a representative 
% value for $p$ from each hypothesis interval and create a discrete prior distribution 
% reflecting our ignorance of the correct hypothesis.

p=[.1 .3 .5 .7 .9];
prior=[.2 .2 .2 .2 .2];
%% 
% The only way we can improve it is to collect and analyze some data. Suppose 
% we go out on three consecutive days, capture 

n=20;
%% 
% birds (with replacement) from the population, and make note of how many of 
% them showed symptoms of heavy metal contamination. The data we collect takes 
% the following form:

D=[7 10 9]; %numbers of contaminated birds per sample.
%% 
% We begin to analyze only the first day of data in order to refine our understanding 
% of which hypothesis is more likely to be correct. Compute a likelihood function 
% for this data ($F$), a posterior likelihood function ($G$), a marginal probability, 
% and a posterior probability. 

F=binopdf(D(1),n,p);
G=F.*prior;
marg=sum(G);
posterior=G/marg
%% 
% Which of the posterior probabilities is the greatest? This corresponds to 
% the hypothesis that is the most likely to be true, given the analysis we've 
% performed for the first day of data.
%% 
% Update the prior to the most recently computed posterior and repeat the process 
% for the first _two_ days worth of data.

prior=posterior;
F=binopdf(D(1),n,p).*binopdf(D(2),n,p)
G=F.*prior
marg=sum(G)
posterior=G/marg
%% 
% Which of the posterior probabilities is the greatest? This corresponds to 
% the hypothesis that is the most likely to be true, given the analysis we've 
% performed for the first two days of data
%% 
% Update the prior to the most recently computed posterior and repeat the process 
% for the first _three_ days worth of data.

prior=posterior;
F=binopdf(D(1),n,p).*binopdf(D(2),n,p).*binopdf(D(3),n,p)
G=F.*prior
marg=sum(G)
posterior=G/marg

pEst=dot(p,posterior)