#!/bin/bash

dataset_root_path="/mnt/HDD-1/harsh/datasets/imdb_wiki/output"

for algo in GroupTestingSumEigen GroupTestingSumClasswise DoubleGroupTestingSumEigen DoubleGroupTestingSumClasswise; do
    echo -n > results/$algo/imdb_wiki_rho0.800000/agg.txt;
    done

for i in {1..5}; do
    taskset -c 5 ./gtnn_sum -n imdb_wiki -q $dataset_root_path/Q.txt -d $dataset_root_path/X.txt -r ./results -R 0.8 -N 520458 -Q 10000 -D 1000
    taskset -c 5 ./dbgtnn_sum -n imdb_wiki -q $dataset_root_path/Q.txt -d $dataset_root_path/X.txt -r ./results -R 0.8 -N 520458 -Q 10000 -D 1000
    taskset -c 5 ./classwise_dbgtnn_sum -n imdb_wiki -q $dataset_root_path/Q.txt -d $dataset_root_path/X.txt -r ./results -R 0.8 -N 520458 -Q 10000 -D 1000
    taskset -c 5 ./classwise_gtnn_sum -n imdb_wiki -q $dataset_root_path/Q.txt -d $dataset_root_path/X.txt -r ./results -R 0.8 -N 520458 -Q 10000 -D 1000
    done
