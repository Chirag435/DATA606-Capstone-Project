import streamlit as st
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import yfinance as yf
import plotly.graph_objects as go
#from datetime import date
import datetime as dt
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#import seaborn as sns
import riskfolio as rp
import plotly.express as px

# Function to calculate portfolio returns
@st.cache
def calculate_portfolio_returns(data, weights):
    pct_returns = data.pct_change()
    portfolio_returns = pct_returns.dot(weights)
    return portfolio_returns

# Function to calculate total invested money
@st.cache
def calculate_total_invested_money(amount, weights):
    total_invested_money = amount * sum(weights)
    return total_invested_money

# Function to calculate volatility for each stock
@st.cache
def calculate_rolling_volatility(pct_returns):
    rolling_volatility = pct_returns.rolling(window=30).std() * (30 ** 0.5)
    return rolling_volatility

# Function to predict prices for the next 30 days
@st.cache
def predict_prices(data):
    # Exclude the 'Symbol' column
    data = data.drop('Symbol', axis=1)

    # Split the data into features (X) and target variable (y)
    X = data.drop('Close', axis=1)
    y = data['Close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict the prices
    predicted_prices = model.predict(X_test)

    return predicted_prices


@st.cache
def get_historical_data(selected_stocks, sdate, edate):
    historical_data = pd.DataFrame()

    for symbol in selected_stocks:
        symbol_data = yf.download(symbol, start=sdate, end=edate, progress=False)
        symbol_data['Symbol'] = symbol
        historical_data = historical_data.append(symbol_data)

    return historical_data



# Setting page configuration and app title
st.set_page_config(page_title="Portfolio Analysis Dashboard", page_icon=":bar_chart:", layout="wide")

# def add_bg_from_url():
#     st.markdown(
#          f"""
#          <style>
#          .stApp {{
#              background-image: url("https://wallpaperaccess.com/full/1393764.png");
#              background-attachment: fixed;
#              background-size: cover
#          }}
#          </style>
#          """,
#          unsafe_allow_html=True
#      )
# add_bg_from_url() 

# Title and Description of the app
st.title('Portfolio Performance Analysis Dashboard')
st.image("https://thumbor.forbes.com/thumbor/fit-in/900x510/https://www.forbes.com/advisor/wp-content/uploads/2021/09/how-to-make-money-in-stocks.jpg")
st.markdown('DATA 606: Capstone Project by Harshil and Chirag')
st.markdown('This project takes user input for selecting the stocks and stock market index for their desired date to provide the analysis on the asset investment in the portfolio.')

yf.pdr_override()

# Predefined start date, later it will be changed by the user
sdate = '2019-01-01'
# Predefined end date, later it will be changed by the user
edate = '2023-01-01'

# Creating a list for stock indexes to select from
stock_indices = ['^GSPC', '^DJI', '^IXIC', '^NYA', '^RUT']

# Using while-True to prevent wrong user inputs:

selected_stocks = st.text_input("Please enter the ticker symbols (comma-separated) for the stocks you want to include in your portfolio:")
selected_stocks = [s.strip() for s in selected_stocks.split(',')]

# Select the start and end date
#sdate = st.text_input("Please enter the start date (YYYY-MM-DD) for your portfolio:")
#edate = st.text_input("Please enter the end date (YYYY-MM-DD) for your portfolio:")
sdate = st.date_input("Please enter the start date for your portfolio:", value=dt.date(2019,1,1))
edate = st.date_input("Please enter the end date for your portfolio:", value=dt.date(2023,1,1))

# Ask the user to select an index from the list
selection = st.selectbox("Select a stock index:", stock_indices)

while True:
    try:
        # The user needs to input a preferred stock symbol; this symbol will be saved in the variable "symbol"
        
        # selected_stocks = st.text_input("Please enter the ticker symbols (comma-separated) for the stocks you want to include in your portfolio:")
        # selected_stocks = [s.strip() for s in selected_stocks.split(',')]

        # # Select the start and end date
        # #sdate = st.text_input("Please enter the start date (YYYY-MM-DD) for your portfolio:")
        # #edate = st.text_input("Please enter the end date (YYYY-MM-DD) for your portfolio:")
        # sdate = st.date_input("Please enter the start date for your portfolio:", value=dt.date(2019,1,1))
        # edate = st.date_input("Please enter the end date for your portfolio:", value=dt.date(2023,1,1))

        # # Ask the user to select an index from the list
        # selection = st.selectbox("Select a stock index:", stock_indices)

        # Download the stock data
        stock_data = yf.download(selected_stocks, start=sdate, end=edate)['Adj Close'].dropna()
        stock_data = pd.DataFrame(stock_data)
        index_data = yf.download(selection, start=sdate, end=edate)['Adj Close'].dropna()
        break
    # If some errors occur (e.g., invalid stock symbol), the user gets an information that the input is invalid and must enter something else
    except (KeyError, OSError):
        st.write(f"{selected_stocks} is not a valid stock symbol. Please try again...")

# Ask the user for the amount of money to be invested
amount = st.number_input("Enter the amount of money to be invested:", value=1000.0)

# Creating a variable for equal weights
equal_weight = 1 / len(selected_stocks)

# Ask the user whether to assign weights manually or use equal weights
choice = st.selectbox("Do you want to assign weights manually?", ("Yes", "No"))

if choice.lower() == "yes":
    # Ask the user to enter the weights for each stock
    while True:
        weights = []
        for stock in selected_stocks:
            weight = st.number_input(f"Enter weight for {stock} in decimal (e.g., 0.2, 0.3):")
            weights.append(weight)

        # Normalize the weights to add up to 1
        total_weight = sum(weights)
        if abs(total_weight - 1) < 1e-6:
            break
        else:
            st.write("Error: The sum of weights must be equal to 1. Please try again.")
else:
    # Assign equal weights to all stocks
    weights = [equal_weight] * len(selected_stocks)

# Calculate the amount to be invested in each stock
invested_money = [amount * weight for weight in weights]

# Calculate daily portfolio values
daily_portfolio_values = (stock_data * weights).sum(axis=1)

# Calculate total invested money
total_invested_money = amount * sum(weights)

# Filter the daily portfolio values by the selected date range
portfolio_values = daily_portfolio_values.loc[sdate:edate]

#amount = st.number_input("Amount of money to be invested", value=amount, min_value=0.01)
# weights_type = st.selectbox("Weight assignment type", ["Equal Weights", "Manual Weights"])

# if weights_type == "Manual Weights":
#     weights = []
#     for stock in selected_stocks:
#         weight = st.number_input(f"Enter weight for {stock} in decimal", value=equal_weight, min_value=0.0, max_value=1.0)
#         weights.append(weight)
# else:
#     weights = [equal_weight] * len(selected_stocks)

# Calculate the amount to be invested in each stock
invested_money = [amount * weight for weight in weights]

# Calculate daily portfolio values
daily_portfolio_values = (stock_data * weights).sum(axis=1)

# Calculate total invested money
total_invested_money = amount * sum(weights)

# Filter the daily portfolio values by the selected date range
portfolio_values = daily_portfolio_values.loc[sdate:edate]


col1, col2, col3 = st.columns(3)
with col1:
    button1 = st.button("Asset Comparison")
with col2:
    button2 = st.button("Portfolio Performance")
with col3:
    button3 = st.button("Price Prediction")


# Asset Comparison Section

if button1:
    st.write("## Asset Comparison")
    
    # Download the stock data
    stock_data = yf.download(selected_stocks, start=sdate, end=edate)['Adj Close']
    
    # Create traces for each stock
    traces = []
    for stock in stock_data.columns:
        traces.append(go.Scatter(
            x=stock_data.index,
            y=stock_data[stock],
            mode='lines',
            name=stock
        ))

    # Create the layout for the plot
    layout = go.Layout(
        title=f'Stock Prices from {sdate} to {edate}',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Adjusted Closing Price in USD'),
        xaxis_tickfont_size=14,
        yaxis_tickfont_size=14
    )

    # Create a line chart using Plotly
    fig = go.Figure(data=traces, layout=layout)

    # Display the plot
    st.plotly_chart(fig)

    # Let's calculate the return of the stocks
    returns = stock_data.pct_change()
    returns.dropna(inplace=True)
    pct_returns = returns * 100

    # Calculate return of the stock index
    index_return = index_data.pct_change().dropna()
    pct_index_return = index_return * 100

    # Show percentage return of each stock from the sdate to edate
    pct_change = (stock_data.iloc[-1] / stock_data.iloc[0] - 1) * 100
    st.write("Asset Returns in Portfolio from", sdate, "to", edate, "is:")
    for stock, pct_change in pct_change.items():
        st.write(f"{stock}: {pct_change:.2f}%")

    # Show correlation between the assets in the portfolio
    corr_mat = returns.corr().round(2)
    st.write("Correlation Matrix:")
    st.dataframe(corr_mat)


    # Calculate volatility for each stock
    #rolling_volatility = pct_returns.rolling(window=30).std() * (30 ** 0.5)
    rolling_volatility = pct_returns[selected_stocks].rolling(window=30).std() * (30 ** 0.5)
    rolling_volatility.dropna(inplace=True)
    fig2 = go.Figure()
    #fig2.add_trace(go.Scatter(x=rolling_volatility.index, y=rolling_volatility, mode='lines', name=selected_stocks,
    #                     hovertemplate='<br>Date: %{x}<br>'+ selected_stocks + ': %{y:.2f}%<extra></extra>'))
    for stock in selected_stocks:
        fig2.add_trace(go.Scatter(x=rolling_volatility.index, y=rolling_volatility[stock], mode='lines', name=stock,
                                hovertemplate='<br>Date: %{x}<br>' + stock + ': %{y:.2f}%<extra></extra>'))

    fig2.update_layout(title="Volatility Comparison",
                    xaxis_title='Date',
                    yaxis_title='% Volatility')

    st.plotly_chart(fig2)

    # Calculate the running maximum of the daily stock price
    running_maximum = stock_data.rolling(30, min_periods=1).max()

    # Calculate the drawdown as the percentage decline from the running maximum
    daily_drawdown = stock_data / running_maximum - 1.0

    # Calculate the rolling maximum drawdown
    max_daily_drawdown = daily_drawdown.rolling(30, min_periods=1).min() * 100

    # Drop any rows with missing values
    max_daily_drawdown.dropna(inplace=True)

    # Plot the daily drawdown of each stock using Plotly
    fig3 = go.Figure()
    for stock in max_daily_drawdown.columns:
        fig3.add_trace(go.Scatter(
            x=max_daily_drawdown.index,
            y=max_daily_drawdown[stock],
            mode='lines',
            name=stock,
            hovertemplate='<br>Date: %{x}<br>'+ stock + ': %{y:.2f}%<extra></extra>'
        ))

    fig3.update_layout(
        title="Drawdown Comparison",
        xaxis_title='Date',
        yaxis_title='% Drawdown'
    )

    st.plotly_chart(fig3)


# Portfolio Performance Section

# Create a pandas dataframe with the weights and invested money for each stock
w = pd.DataFrame(list(zip(weights, invested_money)), index=selected_stocks, columns=["Weight", "Invested Money"])

# Normalize the weights to add up to 1
total_weight = sum(weights)
weights = [weight / total_weight for weight in weights]


if button2:
    # Print the weights assigned to each stock
    st.write("\nAsset Allocation\n")
    for i in range(len(selected_stocks)):
        st.write(f"{selected_stocks[i]}: {weights[i]:.2%} ({invested_money[i]:.2f} USD)")

    # Calculate the percentage change in the portfolio values since start date
    start_value = portfolio_values.iloc[0]
    portfolio_pct_change = ((portfolio_values - start_value) / start_value) * 100
    # Calculate the total return of the portfolio
    total_return = (portfolio_values.iloc[-1] - portfolio_values.iloc[0]) / portfolio_values.iloc[0]

    # Calculate the total value of the portfolio if $X was invested
    portfolio_value = (total_return + 1) * amount

    # Calculate the net profit or loss in dollars and percentage
    net_profit_loss = portfolio_value - total_invested_money
    net_profit_loss_pct = net_profit_loss / total_invested_money

    # Display the results to the user
    #st.write(f"\nYour investment of ${amount:.2f} on {sdate} would be worth ${portfolio_value:.2f} today in your Portfolio.")
    st.markdown(f"<span style='font-family: Arial; font-size: 16px;'>\nYour investment of ${amount:.2f} on {sdate} would be worth ${portfolio_value:.2f} today in your Portfolio.</span>", unsafe_allow_html=True)
    #st.write(f"\nYour investment of ", amount, " on ", {sdate} ,"be worth ", portfolio_value, " today in your Portfolio.")
    if net_profit_loss > 0:
        st.markdown(f"You would have made a profit of ${net_profit_loss:.2f} ({net_profit_loss_pct:.2%}).")
        #st.write(f"You would have made a profit of ", {net_profit_loss:.2f}, "to", edate, "is:")
    else:
        st.markdown(f"You would have incurred a loss of ${abs(net_profit_loss):.2f} ({abs(net_profit_loss_pct):.2%}).")

    # Plot the portfolio returns using Plotly
    returns_trace = go.Scatter(
        x=portfolio_values.index,
        y=portfolio_values.values,
        mode='lines',
        name='Portfolio Returns',
        hovertemplate='<br>Date: %{x}<br>Portfolio Value: $%{y:.2f} (%{text:.2f}%)',
        text=portfolio_pct_change
    )
    # Create a layout for the plot
    layout = go.Layout(
        title='Portfolio Performance',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Portfolio Value')
    )

    # Create the figure and add the trace and layout
    fig = go.Figure(data=[returns_trace], layout=layout)

    # Display the plot
    st.plotly_chart(fig)


    # Calculate the daily returns for the portfolio and benchmark
    portfolio_returns = daily_portfolio_values.pct_change().dropna()
    benchmark_returns = index_data.pct_change().dropna()

    # Calculate the cumulative returns for the portfolio and benchmark
    portfolio_cum_returns = (1 + portfolio_returns).cumprod()
    benchmark_cum_returns = (1 + benchmark_returns).cumprod()

    # Plot the portfolio and benchmark cumulative returns using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_cum_returns.index,
        y=portfolio_cum_returns.values,
        name='Portfolio',
        hovertemplate='Portfolio: $%{y:.2f} (%{text}%)<br>%{x}<extra></extra>',
        text=[f'{p:.2f}' for p in portfolio_returns[-len(portfolio_cum_returns):] * 100]
    ))
    fig.add_trace(go.Scatter(
        x=benchmark_cum_returns.index,
        y=benchmark_cum_returns.values,
        name='Benchmark',
        hovertemplate='Benchmark: $%{y:.2f} (%{text}%)<br>%{x}<extra></extra>',
        text=[f'{b:.2f}' for b in benchmark_returns[-len(benchmark_cum_returns):] * 100]
    ))
    fig.update_layout(
        title='Portfolio Vs Benchmark Performance',
        xaxis_title='Date',
        yaxis_title='Cumulative Return'
    )

    # Display the plot
    st.plotly_chart(fig)

# Price Prediction Section

if button3:
    st.header("Price Prediction")

    # Select the stock for price prediction
    selected_stock = st.selectbox("Select a stock for price prediction", selected_stocks)

    # Download historical stock data for prediction
    stock_df = yf.download(selected_stock, start=sdate, end=edate)

    # Prepare the data for prediction
    prediction_data = stock_df['Adj Close'].values

    # Perform price prediction using your desired algorithm (e.g., ARIMA, LSTM, etc.)
    # Replace the code below with your own price prediction algorithm

    # Calculate the predicted prices for the selected stock
    historical_data = get_historical_data(selected_stocks, sdate, edate)
    selected_stock_data = historical_data[historical_data['Symbol'] == selected_stock]
    predicted_prices = predict_prices(selected_stock_data)

    # Display the predicted prices
    # Display the predicted prices
    st.subheader(f"Predicted {selected_stock} Prices:")
    predicted_prices = [float(price) for price in predicted_prices]
    price_table = pd.DataFrame({"Day": range(1, len(predicted_prices) + 1), "Price": predicted_prices})
    # Display the first 5 rows
    rows_to_display = 5
    st.table(price_table.set_index("Day").head(rows_to_display))

    # Check if there are more rows to display
    if len(predicted_prices) > rows_to_display:
        show_more = st.button("Show more")
        if show_more:
            # Clear the existing content
            st.table(price_table.set_index("Day").iloc[rows_to_display:])

            # Display the remaining rows
            remaining_rows = len(predicted_prices) - rows_to_display
            while remaining_rows > 0:
                rows_to_display = min(remaining_rows, 5)
                placeholder = st.empty()

                # Display the next set of rows
                placeholder.table(price_table.set_index("Day").iloc[rows_to_display:rows_to_display+rows_to_display])

                # Update the remaining rows count
                remaining_rows -= rows_to_display

                # Show "Show more" button for the remaining rows
                if remaining_rows > 0:
                    show_more = placeholder.button("Show more")
                    if not show_more:
                        break

    st.markdown("---")

    # Option to analyze portfolio return
    if st.button("Portfolio Return Analysis"):
        st.header("Portfolio Return Analysis")

        # Create a pandas dataframe with the weights and invested money for each stock
        w = pd.DataFrame(list(zip(weights, invested_money)), index=selected_stocks, columns=["Weight", "Invested Money"])

        # Normalize the weights to add up to 1
        total_weight = sum(weights)
        weights = [weight / total_weight for weight in weights]

        # Display the weights assigned to each stock
        st.subheader("Asset Allocation")
        for i in range(len(selected_stocks)):
            st.write(f"{selected_stocks[i]}: {weights[i]:.2%} ({invested_money[i]:.2f} USD)")

        # Calculate the net profit or loss in dollars and percentage
        net_profit_loss = portfolio_value - total_invested_money
        net_profit_loss_pct = net_profit_loss / total_invested_money

        # Display the results to the user
        st.markdown(f"\nYour investment of ${amount:.2f} on {sdate} would be worth ${portfolio_value:.2f} today in your portfolio.")
        if net_profit_loss > 0:
            st.markdown(f"You would have made a profit of ${net_profit_loss:.2f} ({net_profit_loss_pct:.2%}).")
        else:
            st.markdown(f"You would have incurred a loss of ${abs(net_profit_loss):.2f} ({abs(net_profit_loss_pct):.2%}).")

# End of Streamlit app

