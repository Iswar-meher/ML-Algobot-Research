import yfinance as yf  # Importing the yfinance library

# Fetch stock data for Tata Consultancy Services (TCS)
def fetch_stock_data(symbol):
    # Download stock data from Yahoo Finance for the last 5 years
    df = yf.download(symbol, start="2018-04-07", end="2023-04-07")
    return df

# Main function
def main():
    # Define the stock symbol for TCS (Tata Consultancy Services)
    symbol = 'TCS.NS'  # '.NS' is used for NSE-listed stocks on Yahoo Finance
    
    # Fetch the stock data
    df = fetch_stock_data(symbol)
    
    # Print the first few rows of the data
    print(df.head())  # Display the first 5 rows of the dataset

# Run the script
if __name__ == "__main__":
    main()
