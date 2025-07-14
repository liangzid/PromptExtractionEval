UR_lss = [
    [4.54, 56.28, 13.63],
    [3.18, 4.11, 16.36],
    [11.36, 29.00, 30.00],
    [
        0.90,
        16.87,
        2.27,
    ],
    [0.90, 46.31, 8.63],
]


acc_lss = [
    [82.70, 73.08, 92.52],
    [
        79.68,
        72.40,
        92.50,
    ],
    [81.30, 71.88, 93.14],
    [83.40, 70.00, 89.63],
    [82.15, 69.44, 92.95],
]


def trade_of_m(a, b, lambdaa=0.5):
    return lambdaa * (100 - a) + (1 - lambdaa) * b


lists = []

for i in range(5):
    ls = []
    for j in range(3):
        ls.append(
            trade_of_m(
                UR_lss[i][j],
                acc_lss[i][j],
            )
        )
    lists.append(ls)

print("===========================")
print(lists)
