#!/bin/bash

git pull
git add data/raw_data/ihub/breakout_boards.csv
git add data/raw_data/ihub/most_read.csv
git commit -m 'updated boards'
git push
