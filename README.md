# orthogonal-bandits
Orthogonal Bandit Portfolios 

## Usage

ticker = ['VOW.DE','BA', 'AMD', 'AAPL','GME','CVGW','CAMP','WSCI','LNDC','WOR']  #<- ticker lists    
start_date=[2007, 9, 1]  #<- starting history date   
end_date=[2009,12,9]  #<- ending history date
tau = 300             #<- Length of the rolling window (starting fro, start date and shifting by +1 at each inference step)      
Nsign = 4            #<- number of significant portfolios. To be tuned.      

Inference window is [start_date+tau+1:end_date].   

Cumulative Wealth (CW) comparison among OBP, OBP w/o shortsale and Equally Weighted (EW) portfolios.   
![alt text](https://github.com/pretidav/orthogonal-bandits/raw/main/fig/OBPvsEW.png)  
Theta mixing among relevant and irrelevant orthogonal portfolios (theta=1 all irrelevant, theta=0 all relevant)  
![alt text](https://github.com/pretidav/orthogonal-bandits/raw/main/fig/theta.png)  
Invested percentage for OBP. Negative values correspond to short sale balancing values larger than 100%.    
![alt text](https://github.com/pretidav/orthogonal-bandits/raw/main/fig/weights.png)  
Invested percentage for OBP without short sale possibility. Minimum % is clipped to 0 and all other weights are normalized consequently.  
![alt text](https://github.com/pretidav/orthogonal-bandits/raw/main/fig/weights_nss.png)  


Inspired by "Portfolio Choices with Orthogonal Bandit Learning" W. Shen et al. 2015. 

