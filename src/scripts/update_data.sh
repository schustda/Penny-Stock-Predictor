#!/bin/bash

git pull
git add data/data
git add data/raw_data/stock
git add data/raw_data/ihub/message_boards
git commit -m 'updated data'
git push
