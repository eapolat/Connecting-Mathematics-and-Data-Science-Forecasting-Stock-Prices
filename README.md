# Connecting-Mathematics-and-Data-Science-Forecasting-Stock-Prices

This project, completed as my graduation thesis at Bilkent University (Math 490), explores the intersection of stochastic calculus and machine learning. It provides a comprehensive comparative analysis of four distinct modeling architectures used to forecast stock returns, specifically using daily log-returns of the SPY ETF.

The core objective is to evaluate whether increased model complexity (transitioning from classical probabilistic models to deep learning) translates into superior predictive performance in real-world, noise-heavy financial markets.

The report contains detailed mathematical derivations for each model, including the use of Itô's Lemma to solve the GBM stochastic differential equation. You can see the derivations and proofs step by step. Also contains a side-by-side comparison of models grounded in probability theory (GBM, GARCH) versus statistical and neural network approaches (ARIMA, LSTM). 

All models are trained and tested on real life market data (S&P 500 tracking ETF) which emphasizes practical usability over idealized simulations. 

## Models Covered

+ Geometric Brownian Motion (GBM): Modeling constant drift and volatility through stochastic differential equations.
+ GARCH(1,1): Addressing volatility clustering by modeling time-varying conditional variance.
+ ARIMA: A classical statistical benchmark using autoregressive and moving average components to capture linear dependencies.
+ LSTM Networks: A non-linear, deep learning approach designed to learn complex temporal relationships directly from data.

## Testing With Real World Data

I used daily adjusted closing prices of the SPY ETF, converted them into log-returns for additivity and statistical standard. All models are trained on the first 80% of the historical data and validated against the remaining 20% (Test Set). Each model is tasked with a one-step-ahead prediction, ensuring a fair and identical comparison across all four methods. And their performance is tested using Root Mean Square Error (RMSE), Mean Absolute Error (MAE), and Directional Accuracy to assess both magnitude and trend prediction.

The included report features extensive analysis and deep comments on model performances, and it discusses why certain complex architectures may or may not outperform classical methods in specific financial contexts.

There are still a lof of details and explanations in the report. You can see both the report and the code that I wrote for this project in the folders.

If you have any further questions about the project, you can contact me by: e.anil.polat@gmail.com or https://www.linkedin.com/in/emre-an%C4%B1l-polat-7b05b2262/ 
