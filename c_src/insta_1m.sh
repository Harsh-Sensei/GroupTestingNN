#!/bin/bash

dataset_root_path="/mnt/HDD-1/harsh/datasets/insta_1m/output"

for algo in GroupTestingSumEigen GroupTestingSumClasswise DoubleGroupTestingSumEigen DoubleGroupTestingSumClasswise; do
    echo -n > results/$algo/insta_1m_rho0.800000/agg.txt;
    done

for i in {1..5}; do
    taskset -c 4 ./gtnn_sum -n insta_1m -q $dataset_root_path/Q.txt -d $dataset_root_path/X.txt -r ./results -R 0.8 -N 999991 -Q 10000 -D 1000
    taskset -c 4 ./dbgtnn_sum -n insta_1m -q $dataset_root_path/Q.txt -d $dataset_root_path/X.txt -r ./results -R 0.8 -N 999991 -Q 10000 -D 1000
    taskset -c 4 ./classwise_dbgtnn_sum -n insta_1m -q $dataset_root_path/Q.txt -d $dataset_root_path/X.txt -r ./results -R 0.8 -N 999991 -Q 10000 -D 1000
    taskset -c 4 ./classwise_gtnn_sum -n insta_1m -q $dataset_root_path/Q.txt -d $dataset_root_path/X.txt -r ./results -R 0.8 -N 999991 -Q 10000 -D 1000
    done
