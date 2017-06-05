# Penny-Stock-Predictor

### GOAL:
Predict a massive change in stock price for a given small cap stock
### DATA SETS :
* Web-scraped data from http://investorshub.advfn.com/
* Past stock prices and volume

### MY APPROACH:
* Web scrape data from iHub
* Compile with historical stock data
* Create algorithm to determine success for target

### MODEL
* Feature Space:
    * 52-week prior stock volume
    * 52-week prior iHub message frequency
    * Others (promotional emails, iHub’s “top boards”)
* Target:
    * ‘buy’ (1) : a significant stock price change is upcoming
    * ‘no buy’ (0): no significant stock price change is upcoming





##### Data Folder Structure

```
-- data

    -- raw_data
        -- ihub
            -- message_boards
            -- top_boards
            -- breakout_boards
        -- stock
            -- {raw stock data}

    -- data
        -- compiled data
```

##### SRC Folder Structure

```
-- src

    -- data_management
        -- ihub_data.py
            INPUT: Ticker Symbol
            OUTPUT: create/update the data/raw_data/ihub folder
        -- stock_data.py
            INPUT: Ticker Symbol
            OUTPUT: create/update the data/raw_data/stock folder
        -- compile_data.py
            INPUT: None
            OUTPUT: combined/manipulated data

    -- model
        -- model.py
            INPUT: data from the model_data folder
            OUTPUT: model

    -- data_visualization
        -- tbd

    -- web_app
        -- tbd

```
