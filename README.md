# Penny-Stock-Predictor

FINAL MODEL SCORE ON HOLDOUT SET (XGBoost)
* AUC: 0.89
* Recall: 0.74
* Precision: 0.66



Tasks:
* Refine the target and make sure there are no outliers - COMPLETE
* Begin running top_board / breakout_boards updater - COMPLETE
* Re-visit target definition
* Add top boards within the compiler - not till a later date...
* Create log for AWS instance to update
*




Questions:

Q: How do I choosed which points to select for the training data?

A: Have three options remaining. Choose all points, or have a percentage
of possible points


Q: How do I treat weekends and market holidays?

A: Remove them. The message boards are much less active on those days.


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
    * prior stock volume
    * prior iHub message frequency
    * iHub 'Most Read'
    * iHub 'Breakout Boards'
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


Sources:

http://www.marcoaltini.com/blog/dealing-with-imbalanced-data-undersampling-oversampling-and-proper-cross-validation
