import numpy as np
import os
import h5py
from densityPeakRNN import densityPeakRNN
from cluster import series_to_centers
from update_RNNpeak import update_peak
from evaluation import RandIndex
from evaluation import Purity
from evaluation import NMI
from save_to_excel import write_excel_xlsx

class PSO(object):
    def __init__(self, K,R,population_size,max_step, y,single_distance_between,center_label): # W,
        self.population_size = population_size
        self.K = K
        self.R = R
        self.y = y
        self.w = 0.6
        self.max_step = max_step
        self.c1 = self.c2 = 2 # 学习因子
        self.v = np.random.rand(self.population_size, self.K, self.R)
        self.single_distance_between = single_distance_between
        self.x = np.random.rand(self.population_size, self.K, self.R)  # 初始化粒子群位置;即权重
        for p in range(self.population_size):
            for k in range(self.K):
                sumkr = np.sum(self.x[p][k, :])
                self.x[p][k, :] = self.x[p][k, :] / sumkr
        self.x[0] = np.ones((K, R))
        self.x[0] = self.x[0] / R
        self.center_label = center_label
        fitness,self.part = self.calculate_fitness(self.x,single_distance_between,self.center_label)
        self.p = self.x  # 个体的最佳位置
        self.pg = self.x[np.argmin(fitness)]  # 全局最佳位置
        self.individual_best_fitness = fitness  # 个体的最优适应度
        self.global_best_fitness = np.min(fitness)  # 全局最佳适应度
        self.best_part = self.part[np.argmin(fitness)]
        self.local_best_parts = self.part
        self.all_evals = []
        self.all_evals.append(['RI', 'Purity', 'NMI','global_best_fitness','mean_fitness'])
        temp = []
        RI_value = RandIndex(self.best_part, y)
        purity_value = Purity(self.best_part, y)
        NMI_value = NMI(self.best_part, y)
        temp.append(RI_value)
        temp.append(purity_value)
        temp.append(NMI_value)
        temp.append(self.global_best_fitness)
        temp.append(np.mean(fitness))
        self.all_evals.append(temp)

        print("初始RandIndex：%.4f,PU:%.4f" % (RI_value,purity_value))
        print('best fitness: %.5f, mean fitness: %.5f' % (self.global_best_fitness, np.mean(fitness)))

    # 更新粒子的位置（即子空间权重）
    def update_W(self,C, IDX, W,single_distance_between):
        K = len(C)
        N = len(IDX)
        R = single_distance_between.shape[2]
        in_cluster = [[] for _ in range(K)]
        for n in range(N):
            in_cluster[int(IDX[n])].append(n)
        out_cluster = []
        for i in range(K):
            temp = []
            for j in range(K):
                if j != i:
                    temp = temp + in_cluster[j]
            out_cluster.append(temp)

        HA = np.zeros((K, R))
        MMD = np.zeros((K, R))
        pkr = np.zeros((K, R))

        for k in range(K):
            length_ck = len(in_cluster[k])
            length_nk = len(out_cluster[k])
            if length_ck > 1:
                for r in range(R):
                    interdistance = np.zeros(length_ck * (length_ck - 1) // 2)
                    outsidedistance = np.zeros(length_nk * (length_nk - 1) // 2)

                    counter_i = 0
                    for i in range(length_ck):
                        for j in range(i + 1, length_ck):
                            interdistance[counter_i] = single_distance_between[in_cluster[k][i], in_cluster[k][j], r]
                            counter_i += 1

                    counter_i = 0
                    for i in range(length_nk):
                        for j in range(i + 1, length_nk):
                            outsidedistance[counter_i] = single_distance_between[
                                out_cluster[k][i], out_cluster[k][j], r]
                            counter_i += 1

                    mean_distance_between = np.mean(interdistance)
                    std_distance_between = np.std(interdistance)
                    ukr2 = np.mean(outsidedistance)
                    skr2 = np.std(outsidedistance)
                    HD = np.sqrt(
                        1 - np.sqrt((2 * std_distance_between * skr2) / (std_distance_between ** 2 + skr2 ** 2)) \
                        * np.exp(-(mean_distance_between - ukr2) ** 2 / (4 * (std_distance_between ** 2 + skr2 ** 2))))


                    akr = 0
                    for i in in_cluster[k]:
                        akr = akr + np.power(single_distance_between[i, int(C[k]), r], 2)

                    akr = np.sqrt(akr) / length_ck  # akr为0

                    MMD[k, r] = HD
                    pkr[k, r] = akr
                    HA[k, r] = MMD[k, r] / pkr[k, r]

                    # lamda = 0.5
                    # HA[k, r] = lamda * W[k, r] + (1 - lamda) * HA[k, r]

                HA[np.isnan(HA)] = 1 / R
                sumkr = np.sum(HA[k, :])
                if sumkr != 0:
                    HA[k, :] = HA[k, :] / sumkr
                else:
                    HA[k, :] = 1 / R
            else:
                HA[k, :] = 1 / R

        # # 去冗余
        # corr = np.around(np.corrcoef(HA.T), decimals=4)
        # # 每一行最大值
        # for k in range(K):
        #     best_r = np.argmax(HA[k, :])
        #     for r in range(R):
        #         HA[k, r] = abs(corr[best_r, r]) * HA[k, r]
        # for k in range(K):
        #     sumkr = np.sum(HA[k, :])
        #     HA[k, :] = HA[k, :] / sumkr
        return HA

    def subspace_evaluation6(self,in_cluster,center_label,W,single_distance_between):
        K = W.shape[0]
        R = W.shape[1]
        N = single_distance_between.shape[0]
        # 类间
        inter_cluster = float("inf")
        for i in range(1,len(center_label)):
            for j in range(i):
                temp = 0
                for r in range(R):
                    temp += W[i,r] * np.power(single_distance_between[i, j, r], 2) + W[j,r] * np.power(single_distance_between[i, j, r], 2)
                temp = np.sqrt(temp)
                if temp < inter_cluster:
                    inter_cluster = temp
        # 类内
        intra_cluster = 0
        for k in range(K):
            length_ck = len(in_cluster[k])
            c = int(center_label[k])
            temp = 0
            for n in in_cluster[k]:
                for r in range(R):
                    temp += np.power(single_distance_between[c, n, r], 2) * W[k, r]
            temp = np.sqrt(temp) / length_ck
            intra_cluster += temp

        cost = intra_cluster /(N * inter_cluster)

        # for k in range(K):
        #     for r in range(R):
        #         cost += W[k, r] * np.log(W[k, r])
        return cost

    # 计算粒子的适应值
    def calculate_fitness(self, x,single_distance_between,center_labels):
        F = np.zeros(self.population_size)
        N = single_distance_between.shape[0]
        P_part = np.zeros((self.population_size,N))
        for p in range(self.population_size):
            # 1、权重
            W = x[p]
            # 2、划分矩阵
            in_cluster = [[] for _ in range(self.K)]
            for n in range(N):
                ck = series_to_centers(single_distance_between, n, center_labels[p], W)
                P_part[p, n] = ck
                in_cluster[ck].append(n)

            out_cluster = []
            for i in range(self.K):
                temp = []
                for j in range(self.K):
                    if j != i:
                        temp = temp + in_cluster[j]
                out_cluster.append(temp)
            # fitness
            # fitness = subspace_evaluation(in_cluster,out_cluster,center_labels[p],single_distance_between)
            # fitness = subspace_evaluation3(in_cluster, center_labels[p], W, single_distance_between)
            # fitness = subspace_evaluation2(in_cluster,out_cluster,center_labels[p],W,single_distance_between)
            # fitness = subspace_evaluation4(in_cluster,out_cluster,center_labels[p],W,single_distance_between)
            fitness = self.subspace_evaluation6(in_cluster,center_labels[p],W,single_distance_between)
            F[p] = fitness
        return F,P_part

    def evolve(self):
        # iter = 0
        # while iter < self.max_step - 1:
        #     iter += 1
        best = float("inf")
        while best != self.global_best_fitness:
            best = self.global_best_fitness

            # 更新聚类中心
            centers = self.center_label[0]
            centers = update_peak(self.best_part, centers, self.pg, self.single_distance_between)
            for p in range(self.population_size):
                self.center_label[p] = centers

            # r1 = np.random.rand(self.population_size, self.K, self.R)  # 0-1随机数;
            # r2 = np.random.rand(self.population_size, self.K, self.R)
            # # 更新速度和权重;向量计算
            # self.v = self.w * self.v + self.c1 * r1 * (self.p - self.x) + self.c2 * r2 * (self.pg - self.x)
            # self.x = self.v + self.x
            # for p in range(self.population_size):
            #     for k in range(self.K):
            #         sumkr = np.sum(self.x[p][k, :])
            #         self.x[p][k, :] = self.x[p][k, :] / sumkr

            r = np.random.uniform(0,1)
            C = np.zeros(np.shape(self.pg))
            for i in range(self.population_size):
                self.x[i] = r * self.p[i] + (1-r) * self.pg  # X(g)
                C = C + self.p[i]
            C = C / self.population_size
            alpha = 0.6
            a = np.random.randint(0,2)
            temp = [-1,1]
            # X(g+1)
            self.x = self.x + temp[a] * alpha * abs(C - self.x) * np.log(1/np.random.uniform(0,1))
            for p in range(self.population_size):
                for k in range(self.K):
                    sumkr = np.sum(self.x[p][k, :])
                    self.x[p][k, :] = self.x[p][k, :] / sumkr

            # for iter in range(self.max_step - 1):
            #     for i in range(self.population_size):
            #         # 更新每个粒子的位置（权重）
            #         self.x[i] = self.update_W(self.center_label[i], self.part[i],self.x[i],self.single_distance_between)
            #         # 不用更新后的权重？
            #         self.center_label[i] = update_peak(self.part[i], self.center_label[i], self.x[i], self.single_distance_between)

            fitness,self.part = self.calculate_fitness(self.x,self.single_distance_between,self.center_label)

            # 需要更新的个体
            update_id = np.greater(self.individual_best_fitness, fitness)
            self.p[update_id] = self.x[update_id]
            self.local_best_parts[update_id] = self.part[update_id]
            self.individual_best_fitness[update_id] = fitness[update_id]
            # 新一代出现了更小的fitness，所以更新全局最优fitness和位置
            if np.min(self.individual_best_fitness) < self.global_best_fitness:
                self.pg = self.p[np.argmin(self.individual_best_fitness)]  # 更新全局最佳位置
                self.global_best_fitness = np.min(self.individual_best_fitness)  # 更新全局最佳适应度
                self.best_part = self.local_best_parts[np.argmin(self.individual_best_fitness)]

            temp = []
            RI_value = RandIndex(self.best_part, self.y)
            purity_value = Purity(self.best_part, self.y)
            NMI_value = NMI(self.best_part, self.y)
            temp.append(RI_value)
            temp.append(purity_value)
            temp.append(NMI_value)
            temp.append(self.global_best_fitness)
            temp.append(np.mean(fitness))
            self.all_evals.append(temp)

            print("RandIndex：%.4f,PU:%.4f" % (RI_value, purity_value))
            print('best fitness: %.5f, mean fitness: %.5f' % (self.global_best_fitness, np.mean(fitness)))

        return self.all_evals # self.best_part

def main():
    choice = 1
    if choice == 1:
        files = ['ArticularyWord'] # 'lp5','cricket','ArticularyWord','uWaveGesture','net534','ArticularyWord','BCI'
    else:
        files = ['wafer298'] #'char300','vowels','wafer298',,'WalkvsRun','Wafer','PEMS','NetFlow','KickvsPunch','ECG','JapaneseVowels','CMUsubject16','AUSLAN','ArabicDigits'
    for file in files:
        if choice == 1:
            filename = file + '.h5'
            filename = os.path.join("../data/", filename)
            f = h5py.File(filename, 'r')
            # X = f['train_x'][:]
            y = f['train_y'][:]
        else:
            filename = file + '_labels.npy'
            filename = os.path.join("../result/", filename)
            y = np.load(filename)

        print("%s数据集进行子空间聚类..." % file)
        # print('数据集大小', X.shape)

        PATH = '../result/' + file + '_dist.npy'
        single_distance_between = np.load(PATH)
        K = len(np.unique(y))
        print('聚类簇数：', K)
        R = single_distance_between.shape[2]

        # 1、初始化峰值
        k = 5
        center_label = densityPeakRNN(k, K, single_distance_between)
        population_size = 50
        center_labels = np.zeros((population_size,len(center_label)))
        for i in range(population_size):
            center_labels[i] = center_label

        max_steps = 10
        all_evals = []
        print("single_distance_between.shape",single_distance_between.shape)
        for step in range(10):
            print(step)
            pso = PSO(K,R,population_size,max_steps, y,single_distance_between,center_labels)
            if step == 0:
                all_evals = pso.evolve()
            else:
                temp = pso.evolve()
                all_evals.append(temp[1])

        eval_file = file + "_POS_HA"
        book_name_xlsx = '../result/' + file + '_evaluation.xlsx'
        sheet_name_xlsx = eval_file
        write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, all_evals)

if __name__ == '__main__':
    main()