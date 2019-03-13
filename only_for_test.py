import xlrd
import math
import numpy as np
import matplotlib.pyplot as plt

data = xlrd.open_workbook('/home/dl/cf/test data.xlsx')
table = data.sheets()[0]

row = 50
col = 70

def step_one():
    max = []
    min = []
    sed_max = []
    sed_min = []
    avg = []

    for i in np.arange(1, 1 + row):
        a = table.row_values(i, 1)
        a_sort = np.sort(a)
        min.append(a_sort[0])
        max.append(a_sort[-1])
        sed_min.append(a_sort[int(len(a) / 4)])
        sed_max.append(a_sort[int(len(a) * 3 / 4)])
        avg.append(a_sort[int(len(a) / 2)])

    print(np.arange(row), min)
    print(row, len(min))
    x = np.arange(row)
    fig, ax = plt.subplots()
    plt.yscale('symlog')
    ax.plot(x, min, color='darkgrey')
    ax.plot(x, max, color='darkgrey')
    ax.plot(x, sed_min, color='dimgrey')
    ax.plot(x, sed_max, color='dimgrey')
    ax.fill_between(x, min, max, where=max > min, facecolor='silver', label='max-min')
    ax.fill_between(x, sed_min, sed_max, where=sed_max > sed_min, facecolor='dimgrey', label='25%-75% Percentile')
    plt.plot(x, avg, 'gold', label='Mean Value')
    plt.legend(loc=0)
    # plt.show()

def step_two():
    max = []
    min = []
    sed_max = []
    sed_min = []
    avg = []
    a = table.row_values(1, 1)
    table_grd = []
    value = a[np.random.randint(len(a))]
    for i in np.arange(2, 1 + row):
        a = table.row_values(i, 1)
        a_grd = np.array(a) - value
        table_grd.append(a_grd)
        value = a_grd[np.random.randint(len(a))]


    for i in range(len(table_grd)):
        a_sort = np.sort(table_grd[i])
        min.append(a_sort[0])
        max.append(a_sort[-1])
        sed_min.append(a_sort[int(len(a) / 4)])
        sed_max.append(a_sort[int(len(a) * 3 / 4)])
        avg.append(a_sort[int(len(a) / 2)])

    print(np.arange(row), min)
    print(row, len(min))
    x = np.arange(len(table_grd))
    fig, ax = plt.subplots()
    ax.plot(x, min, color='darkgrey')
    ax.plot(x, max, color='darkgrey')
    ax.plot(x, sed_min, color='dimgrey')
    ax.plot(x, sed_max, color='dimgrey')
    ax.fill_between(x, min, max, where=max > min, facecolor='silver', label='max-min')
    ax.fill_between(x, sed_min, sed_max, where=sed_max > sed_min, facecolor='dimgrey', label='25%-75% Percentile')
    plt.plot(x, avg, 'gold', label='Mean Value')
    plt.legend(loc=0)
    plt.show()

if __name__ == '__main__':
    step_one()
    step_two()


