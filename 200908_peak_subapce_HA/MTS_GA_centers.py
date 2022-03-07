import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from cluster import series_to_centers
from densityPeakRNN import densityPeakRNN
from updateW_HA import update_subspace
import math
import random
from evaluation import RandIndex
from evaluation import Purity
from evaluation import NMI
from MTS_GA_W import subspace_evaluation1
from update_feature_weight import update_subspace2
from save_to_excel import write_excel_xlsx


def crossover_and_mutation(pop,K,N,CROSSOVER_RATE=0.8):
    new_pop = []
    for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
        child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
        if np.random.rand() < CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(POP_SIZE)]  # 再种群中选择另一个个体，并将该个体作为母亲
            cross_points1 = np.random.randint(low=0, high=K)  # 随机产生交叉的点
            child[cross_points1:] = mother[cross_points1:]
            # cross_points2 = np.random.randint(low=0, high=N)  # 随机产生交叉的点
            # while cross_points2 == cross_points1:
            #     cross_points2 = np.random.randint(low=0, high=N)  # 随机产生交叉的点
            # if cross_points1 < cross_points2:
            #     child[cross_points1:cross_points2+1] = mother[cross_points1:cross_points2+1]  # 孩子得到位于交叉点后的母亲的基因
            # else:
            #     child[cross_points2:cross_points1+1] = mother[cross_points2:cross_points1+1]  # 孩子得到位于交叉点后的母亲的基因
        mutation(child,K)  # 每个后代有一定的机率发生变异
        new_pop.append(child)

    return new_pop

def mutation(child,K,MUTATION_RATE=0.005):
    N = len(child)
    if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
        mutate_point = np.random.randint(low=0, high=K)  # 随机产生一个实数，代表要变异基因的位置
        child[mutate_point] = np.random.randint(low=0, high=N)

def select(pop, fitness):  # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=(fitness) / (fitness.sum()))
    return pop[idx],idx


def print_info(pop,fitness):
    max_fitness_index = np.argmax(fitness)
    print("max_fitness:", fitness[max_fitness_index])
    part = pop[max_fitness_index]
    RI_value = RandIndex(part, y)
    purity_value = Purity(part, y)
    NMI_value = NMI(part, y)
    print("评价指标",RI_value, purity_value, NMI_value)
    return fitness[max_fitness_index],RI_value,purity_value, NMI_value

def function1(in_cluster,center_label,W,single_distance_between):
    K = len(in_cluster)
    R = single_distance_between.shape[2]
    akr = 0
    for k in range(K):
        if len(in_cluster[k])>1:
            c = center_label[k]
            for x in in_cluster[k]:
                temp = 0
                for r in range(R):
                    # akr
                    temp += np.power(single_distance_between[c,x,r],2) * W[k,r]
                temp = np.sqrt(temp)
                akr += temp
    return akr

def function2(in_cluster,center_label,W,single_distance_between):
    K = len(in_cluster)
    R = single_distance_between.shape[2]
    fitness = 0
    for k in range(K):
        akr = 0
        c = center_label[k]
        for x in in_cluster[k]:
            temp = 0
            for r in range(R):
                # akr
                temp += np.power(single_distance_between[c,x,r],2) * W[k,r]
            temp = np.sqrt(temp)
            akr += temp

        dist = []
        for t in range(K):
            if t != k:
                temp = 0
                for r in range(R):
                    temp += np.power(single_distance_between[k,t,r],2)
                dist.append(np.sqrt(temp))

        bkr = (np.min(dist) + np.mean(dist)) / 2

        fitness += akr / bkr
    return fitness


if __name__ == "__main__":
    path = './data/'
    data_name = ["lp1","lp2","lp3","lp4","lp5"]
    for file in data_name:
        print(file)
        filename = file + '.h5'
        filename = os.path.join(path, filename)
        f = h5py.File(filename, 'r')
        X = f['train_x'][:]
        print(X.shape)
        y = f['train_y'][:]
        K = len(np.unique(y))

        N = len(y)
        POP_SIZE = 50
        CROSSOVER_RATE = 0.8
        MUTATION_RATE = 0.005
        N_GENERATIONS = 50

        all_evals = []

        # 获取正常（没有变量权值）的距离矩阵
        PATH = './result/' + file + '_dist.npy'
        single_distance_between = np.load(PATH)
        R = single_distance_between.shape[2]

        k = 5
        _, special = densityPeakRNN(k, K, single_distance_between)  # return center_label

        g = 0
        for n in range(N):
            temp = 0
            for r in range(R):
                temp += np.power(single_distance_between[special, n, r], 2) * (1 / R)
            temp = np.sqrt(temp)
            g += temp
        g = g / N
        print("g:",g)

        for iter in range(10):
            print("第%d次运行算法" % iter)
            # init
            pop = np.random.randint(K, size=(POP_SIZE, N))
            center_labels = np.zeros((POP_SIZE,K),dtype=int)
            W = np.zeros((POP_SIZE,K,R))
            function1_values = np.zeros(POP_SIZE)
            function2_values = np.zeros(POP_SIZE)

            for p in range(POP_SIZE):
                part = pop[p]
                center_label = np.zeros(K, dtype=int)
                in_cluster = [[] for _ in range(K)]
                for n in range(N):
                    in_cluster[int(part[n])].append(n)
                out_cluster = []
                for i in range(K):
                    temp = []
                    for j in range(K):
                        if j != i:
                            temp = temp + in_cluster[j]
                    out_cluster.append(temp)

                for k in range(K):
                    length_ck = len(in_cluster[k])
                    if length_ck > 1:
                        distance_w = np.zeros((length_ck, length_ck))
                        for i in range(1, length_ck):
                            for j in range(i):
                                for r in range(R):
                                    distance_w[i, j] = distance_w[i, j] + np.power(
                                        single_distance_between[in_cluster[k][i], in_cluster[k][j], r], 2)
                                distance_w[i, j] = np.sqrt(distance_w[i, j])
                                distance_w[j, i] = distance_w[i, j]

                        # minSsumLabel
                        sum_distance = np.sum(distance_w, axis=1)
                        min_dist = float('inf')
                        minSumLabel = -1
                        for i in range(len(sum_distance)):
                            if sum_distance[i] < min_dist:
                                minSumLabel = i  # 0-length_ck
                                min_dist = sum_distance[i]

                        center_label[k] = in_cluster[k][minSumLabel]
                    elif length_ck == 1:
                        center_label[k] = in_cluster[k][0]

                center_labels[p] = center_label

                # init W
                # if p <= POP_SIZE / 2:
                #     W[p] = np.ones((K, R))
                #     W[p] = W[p] / R
                # else:
                W[p], cost = update_subspace2(g, in_cluster, out_cluster, center_label, W[p], single_distance_between)
                # W[p] = update_subspace(g, in_cluster, out_cluster, center_label, W[p], single_distance_between)

                function1_values[p] = subspace_evaluation1(g,in_cluster, out_cluster, center_label, W[p], single_distance_between)
                # function1_values[p] = subspace_evaluation(in_cluster, out_cluster, center_label, single_distance_between)
                # function1_values[p] = function2(in_cluster, center_label, W[p], single_distance_between)
            # function1_values = -(function1_values - np.max(function1_values)) + 1e-3

            max_fitness = float("inf")
            count = 0
            for step in range(N_GENERATIONS):  # 迭代N代
                print(step)
                #筛选
                center_labels,idx = select(center_labels, function1_values)  # 选择生成新的种群
                W = W[idx]
                pop = pop[idx]
                function1_values = function1_values[idx]

                # 交叉变异,生产新群体
                center_labels = np.array(crossover_and_mutation(center_labels,K,N,CROSSOVER_RATE=0.8))
                # centers and W will be updated in the local search procedure
                # local search
                for p in range(POP_SIZE):
                    center_label = center_labels[p]
                    in_cluster = [[] for _ in range(K)]
                    for n in range(N):
                        ck = series_to_centers(single_distance_between, n, center_label, W[p])
                        pop[p][n] = ck
                        in_cluster[ck].append(n)

                    out_cluster = []
                    for i in range(K):
                        temp = []
                        for j in range(K):
                            if j != i:
                                temp = temp + in_cluster[j]
                        out_cluster.append(temp)

                    W[p], cost = update_subspace2(g, in_cluster, out_cluster, center_label, W[p],
                                                  single_distance_between)

                    # W[p] = update_subspace(g, in_cluster, out_cluster, center_label, W[p], single_distance_between)

                    for k in range(K):
                        length_ck = len(in_cluster[k])
                        if length_ck > 1:
                            distance_w = np.zeros((length_ck, length_ck))
                            for i in range(1, length_ck):
                                for j in range(i):
                                    for r in range(R):
                                        distance_w[i, j] = distance_w[i, j] + np.power(
                                            single_distance_between[in_cluster[k][i], in_cluster[k][j], r], 2) # *W[p,k,r]
                                    distance_w[i, j] = np.sqrt(distance_w[i, j])
                                    distance_w[j, i] = distance_w[i, j]

                            # minSsumLabel
                            sum_distance = np.sum(distance_w, axis=1)
                            min_dist = float('inf')
                            minSumLabel = -1
                            for i in range(len(sum_distance)):
                                if sum_distance[i] < min_dist:
                                    minSumLabel = i  # 0-length_ck
                                    min_dist = sum_distance[i]

                            center_label[k] = in_cluster[k][minSumLabel]
                        elif length_ck == 1:
                            center_label[k] = in_cluster[k][0]

                    center_labels[p] = center_label

                    # 3、refine clustering result
                    in_cluster = [[] for _ in range(K)]

                    for n in range(N):
                        ck = series_to_centers(single_distance_between, n, center_label, W[p])
                        pop[p][n] = ck
                        in_cluster[ck].append(n)

                    out_cluster = []
                    for i in range(K):
                        temp = []
                        for j in range(K):
                            if j != i:
                                temp = temp + in_cluster[j]
                        out_cluster.append(temp)

                    function1_values[p] = subspace_evaluation1(g, in_cluster, out_cluster, center_label, W[p],
                                                               single_distance_between)
                    # function1_values[p] = subspace_evaluation(in_cluster, out_cluster, center_label,single_distance_between)
                    # function1_values[p] = function2(in_cluster, center_label, W[p], single_distance_between)

                # 最小化转最大化，且不能小于等于0
                # function1_values = -( function1_values - np.max(function1_values)) + 1e-3
                # print(function1_values.sum())

                temp,RI_value,purity_value, NMI_value = print_info(pop,function1_values)
                if temp == max_fitness:
                    count += 1
                else:
                    count = 1
                    max_fitness = temp
                if count == 3:
                    break

            temp = []
            temp.append(RI_value)
            temp.append(purity_value)
            temp.append(NMI_value)
            all_evals.append(temp)

        print("end")
        print(all_evals)
        # save evaluation
        # eval_file = file + "_MTS_GA_center"
        # book_name_xlsx = './result/' + file + '_MTS_GA_evaluation.xlsx'
        # sheet_name_xlsx = eval_file
        # write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, all_evals)

