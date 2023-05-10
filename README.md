# DATA606-Capstone-Project
## Portfolio Performance Analysis

### Overview:
Portfolio Performance Analysis is a tool created in Python which allows investor to analyse their investment for making better decision on it. This tool
incorporates user input for selecting stocks (Adjsuted closing price), choosing a market index as a benchmark, specifying start and end dates, allocating weights to the asset and determining the amount to be invested. Time series dataset for the selected stockcs, benchmark and T-bill rate are fetched from YahooFinance and FRED API respectively. This tool has three features:
<br> 1. Asset Comparison
<br> 2. Portfolio Perfmance
<br> 3. Forecast Portfolio Return

#### 1. Asset Comparison:
This feature shows comparsion of the all the assest selected by user for a given date range. The comparison includes tracking the historical data of the asset, asset's return correlation, compare volalitility and evaluate drawdown of each asset. Interactive graphs are plotted using plotly to visualise the above analysis.

#### 2. Portfolio Performance:
This feature calculates portfolio performance based on the weights allocated to each asset by the user. If user do not wish to allocate the weights, then automatically equal weights will be distributed across the pool of asset. Also, user is prompted to enter the amount to be invested. Gathering these data from user, portfolio performance is evaluted to show how much profit or loss the investment has experienced. Along with portfolio's performance, benchmark performance is also compared to see where does investor's portfolio return stands.

Risk measures are evaluated by determining the rolling sharpe ratio. This gives investor an idea about the risk of each asset. The risk-free rate is fetched using FRED API to calculate sharpe ratio. Risk-Return trade off graph is plotted for better understanding of risk associated with the return in each asset.

#### 3. Forecast Portfolio Return:
Portfolio's return is forecasted in this feature for the next 30 days horizon. Statistical techniques are used to preprocess the data in ARIMA (Auto Regressive Integrated Moving Average) model. ARIMA model performance well when it comes to forecasting the time series data. It catches the trend and seasonilty in the pattern to make a forecast. The results are evauted using MAE (Mean Absolute Error), MAPE (Mean Absolute Percentage Error), MSE (Mean Squared Error) and RMSE (Root Mean Squared Error) matrices. Along with this, this feature tells investor how much profit or loss will incurr over the forecasted period. See attached image. 
![image](https://github.com/p-harshil/DATA606-Capstone-Project/assets/68314057/b1f3d8b1-c4c6-413d-9617-8c67ac91931a)

### About Dataset:
The time series data will fetched from YahooFinance upon investor's selection of the stock. The tool prompt user to input the ticker symbols, start and end date, benchmark and reads data using yfinance. The T-bill rate is downloaded in the tool from FRED API which is a web services provided by Federal Bank Reserve.

### Front-End:
This Portfolio Performance Analysis tool is productionlized using Streamlit to create the front-end web page. VS Code is used as an editor to create and handle the .py file for front-end.

### Dependencies:
* numpy
* pandas
* plotly
* matplotlib
* statsmodel
* yfinance
* fredapi
* sklealearn
* streamlit

### Future Work:
* At this time we are allowing user to input the weights to allocate to each asset. Another provided option is to distribute is equally. We want to optimize the weights to be allocated to each asset as per their performance.
* Currently we have only deployed univariate analysis in this project. Adding more exogenous variables and performaing multivariate analysis for the model can result in higher accuracy. 

### Instruction:
This project is a part of Capston Study in Data Science graduate program in University of Maryland Baltimore County.
