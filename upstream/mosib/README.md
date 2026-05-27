# Most Similar Biclique Search at Scale

This repository contains the code of our VLDB 2025 paper.

> Most Similar Biclique Search at Scale. Deming Chu, Zhizhi Gao, Fan Zhang, Wenjie Zhang, Xuemin Lin, Zhihong Tian.



## Requirements

- G++
- CMake

## Build


Build with the code below at the project root.

``` shell
mkdir build
cd build
cmake ..
make -j
```

After that, you will get three executable programs.

- `local_exact`: the exact local search algorithm of *Mosib*
- `global_exact`: the exact global search algorithm of *Mosib*
- `global_app`: the approximate global search algorithm of *Mosib-GloApp*

## Run
Build and then run the script at `./build`. By default, we run algorithms on the GitHub dataset (see `./dataset/bi-github.txt`).

1. Run the exact local search on GitHub. The size constraint $\tau$ equals 5. The next 100 integers are the ids of the query vertices.
```
./local_exact ../../dataset/bi-github.txt 5 51 68 94 146 167 197 216 270 271 321 350 375 398 400 406 430 444 445 469 547 551 588 631 645 666 685 690 763 903 915 989 1152 1162 1258 1260 1319 1392 1414 1456 1477 1538 1613 1671 1725 1766 1798 1807 1827 1835 1977 2036 2056 2203 2233 2255 2289 2388 2515 2595 2606 2686 2813 2967 2991 3074 3173 3292 3336 3555 3567 3580 3604 3665 3682 3688 3718 3730 3985 4165 4211 4224 4230 4411 4494 4646 4668 4684 4748 5054 5469 5570 6538 6644 6726 6815 7613 7806 7908 8849 10243
```

2. Run the exact global search on GitHub. The size constraint $\tau$ equals 5.
```
./global_exact ../dataset/bi-github.txt 5
```

3. Run the approximate global search on GitHub. The size constraint $\tau$ equals 5.
```
./global_app ../dataset/bi-github.txt 5
```

## Input File Format

Here is the format of the input file:
```
n_L n_R m
u_1 v_1
.....
u_m v_m
```

The first line contains three integers: the number of left-vertices $n_L$, the number of right-vertices $n_R$, and the number of edges $m$.

The next $m$ lines describe the edges in the graph.
Each line contains the two endpoints $u_i,v_i$ of an edge, where we have $0\leq u_i\lt n_L$ and $0 \leq v_i \lt n_R$.


Please kindly check `./dataset/bi-github.txt` for an example.