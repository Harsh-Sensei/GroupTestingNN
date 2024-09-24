#!/usr/bin/env python3
import csv
out_file = open("aggregate_results.csv", 'w', newline='')
result_file = csv.writer(out_file)
result_file.writerow(["Dataset", "Naive Search (ms)", "Std_dev (ms)", "Grouped dataset (ms)", \
                     "Std_dev (ms)", "Smart grouping (ms)", "Std_dev (ms)",\
                        "Double grouped (ms)", "Std_dev (ms)", "Smart doublegroup (ms)", \
                            "Std_dev (ms)"])

datasets = ["imagenet", "imdb_wiki", "insta_1m", "mirflickr"]
algorithms = ["Naive", "GroupTestingSumEigen", "GroupTestingSumClasswise", \
              "DoubleGroupTestingSumEigen", "DoubleGroupTestingSumClasswise"]

final_results = {}
for dataset in datasets:
    f1 = open("results/GroupTestingSumEigen/" + dataset + "_rho0.800000/agg.txt", 'r')
    f2 = open("results/GroupTestingSumClasswise/" + dataset + "_rho0.800000/agg.txt", 'r')
    f3 = open("results/DoubleGroupTestingSumEigen/" + dataset + "_rho0.800000/agg.txt", 'r')
    f4 = open("results/DoubleGroupTestingSumClasswise/" + dataset + "_rho0.800000/agg.txt", 'r')
    final_results[dataset] = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    f1.readline()
    f2.readline()
    f2.readline()
    f3.readline()
    f3.readline()
    f4.readline()
    f4.readline()
    [num1, num2, num3, num4] = [float(f1.readline().split()[-1]), \
                                    float(f2.readline().split()[-1]), \
                                        float(f3.readline().split()[-1]), \
                                            float(f4.readline().split()[-1])]
    final_results[dataset][1][0] += num1 / 5
    final_results[dataset][1][1] += (num1 ** 2) / 5
    final_results[dataset][2][0] += num2 / 5
    final_results[dataset][2][1] += (num2 ** 2) / 5
    final_results[dataset][3][0] += num3 / 5
    final_results[dataset][3][1] += (num3 ** 2) / 5
    final_results[dataset][4][0] += num4 / 5
    final_results[dataset][4][1] += (num4 ** 2) / 5
    [num1, num2, num3, num4] = [float(f1.readline().split()[-1]), \
                                    float(f2.readline().split()[-1]), \
                                        float(f3.readline().split()[-1]), \
                                            float(f4.readline().split()[-1])]
    final_results[dataset][0][0] += (num1 + num2 + num3 + num4) / 20
    final_results[dataset][0][1] += ((num1 ** 2) + (num2 ** 2) + (num3 ** 2) + (num4 ** 2)) / 20
    for __iter in range (4):
        for __jter in range (5):
            f1.readline()
            f2.readline()
            f3.readline()
            f4.readline()
        [num1, num2, num3, num4] = [float(f1.readline().split()[-1]), \
                                        float(f2.readline().split()[-1]), \
                                            float(f3.readline().split()[-1]), \
                                                float(f4.readline().split()[-1])]
        final_results[dataset][1][0] += num1 / 5
        final_results[dataset][1][1] += (num1 ** 2) / 5
        final_results[dataset][2][0] += num2 / 5
        final_results[dataset][2][1] += (num2 ** 2) / 5
        final_results[dataset][3][0] += num3 / 5
        final_results[dataset][3][1] += (num3 ** 2) / 5
        final_results[dataset][4][0] += num4 / 5
        final_results[dataset][4][1] += (num4 ** 2) / 5
        [num1, num2, num3, num4] = [float(f1.readline().split()[-1]), \
                                        float(f2.readline().split()[-1]), \
                                            float(f3.readline().split()[-1]), \
                                                float(f4.readline().split()[-1])]
        final_results[dataset][0][0] += (num1 + num2 + num3 + num4) / 20
        final_results[dataset][0][1] += ((num1 ** 2) + (num2 ** 2) + (num3 ** 2) + (num4 ** 2)) / 20
    final_results[dataset][0][1] = (final_results[dataset][0][1] - \
                                    final_results[dataset][0][0] ** 2) ** 0.5
    final_results[dataset][1][1] = (final_results[dataset][1][1] - \
                                    final_results[dataset][1][0] ** 2) ** 0.5
    final_results[dataset][2][1] = (final_results[dataset][2][1] - \
                                    final_results[dataset][2][0] ** 2) ** 0.5
    final_results[dataset][3][1] = (final_results[dataset][3][1] - \
                                    final_results[dataset][3][0] ** 2) ** 0.5
    final_results[dataset][4][1] = (final_results[dataset][4][1] - \
                                    final_results[dataset][4][0] ** 2) ** 0.5
    print("Dataset: ", dataset)
    print("Naive search mean time :", final_results[dataset][0][0],\
        "ms, standard deviation", final_results[dataset][0][1], "ms")
    print("Grouping dataset vectors time :", final_results[dataset][1][0],\
          "ms, standard deviation", final_results[dataset][1][1], "ms")
    print("Smart grouping dataset time :", final_results[dataset][2][0],\
          "ms, standard deviation", final_results[dataset][2][1], "ms")
    print("Grouping dataset and query time :", final_results[dataset][3][0],\
          "ms, standard deviation", final_results[dataset][3][1], "ms")
    print("Smart double grouping time :", final_results[dataset][4][0],\
          "ms, standard deviation", final_results[dataset][4][1], "ms")
    print()
    result_file.writerow([dataset, final_results[dataset][0][0], final_results[dataset][0][1], \
                         final_results[dataset][1][0], final_results[dataset][1][1], \
                         final_results[dataset][2][0], final_results[dataset][2][1], \
                         final_results[dataset][3][0], final_results[dataset][3][1], \
                         final_results[dataset][4][0], final_results[dataset][4][1]])
out_file.close()