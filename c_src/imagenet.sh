#!/bin/bash

dataset_root_path="/mnt/HDD-1/harsh/datasets/imagenet/output"

for algo in GroupTestingSumEigen GroupTestingSumClasswise DoubleGroupTestingSumEigen DoubleGroupTestingSumClasswise; do
    echo -n > results/$algo/imagenet_rho0.800000/agg.txt;
    done

for i in {1..5}; do
    taskset -c 7 ./gtnn_sum -n imagenet -q $dataset_root_path/Q.txt -d $dataset_root_path/X.txt -r ./results -R 0.8 -N 1281151 -Q 10000 -D 1000
    taskset -c 7 ./dbgtnn_sum -n imagenet -q $dataset_root_path/Q.txt -d $dataset_root_path/X.txt -r ./results -R 0.8 -N 1281151 -Q 10000 -D 1000
    taskset -c 7 ./classwise_dbgtnn_sum -n imagenet -q $dataset_root_path/Q.txt -d $dataset_root_path/X.txt -r ./results -R 0.8 -N 1281151 -Q 10000 -D 1000
    taskset -c 7 ./classwise_gtnn_sum -n imagenet -q $dataset_root_path/Q.txt -d $dataset_root_path/X.txt -r ./results -R 0.8 -N 1281151 -Q 10000 -D 1000
    done
