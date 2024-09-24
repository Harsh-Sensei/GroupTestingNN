#!/bin/bash
make -B

echo "Starting eigen experiments"

for dataset in imagenet imdb_wiki insta_1m mirflickr; do
    nohup bash ./$dataset.sh &
    done
