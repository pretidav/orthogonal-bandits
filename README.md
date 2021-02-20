# orthogonal-bandits
Orthogonal Bandit Portfolios 

## Usage

ticker = ['VOW.DE','BA', 'AMD', 'AAPL','GME','CVGW','CAMP','WSCI','LNDC','WOR']  #<- ticker lists    
start_date=[2007, 9, 1]  #<- starting history date   
end_date=[2009,12,9]  #<- ending history date
tau = 300             #<- Length of the rolling window (starting fro, start date and shifting by +1 at each inference step)      
Nsign = 4            #<- number of significant portfolios. To be tuned.      

Inference window is [start_date+tau+1:end_date].   

Inspired by "Portfolio Choices with Orthogonal Bandit Learning" W. Shen et al. 2015. 

