import ast
from scipy.stats import ttest_rel, ttest_ind
import numpy as np

with open('stats.txt', 'r') as fp:
    data = fp.read()
    details = ast.literal_eval(data)

methods = ['Without', 'SMOTE', 'Undersampling', 'Oversampling']
stats = ['f1_score', 'precision', 'recall', 'specificity', 'g-mean']

# for stat in stats:
#     print(stat)
#     for method in methods:
#         print(method + str(details[method][stat]))

for stat in stats:
    print(stat)
    for method1 in methods:
        for method2 in methods:
            print(method1 + ' vs ' + method2)
            # print(ttest_rel(details[method1][stat], details[method2][stat]))
            result = ttest_rel(details[method1][stat], details[method2][stat])
            print(str(round(result.statistic, 3)) + '/' + str(round(result.pvalue, 4)))
