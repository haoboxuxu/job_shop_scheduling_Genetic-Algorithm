import math
import pandas as pd
import numpy as np
import time
import copy

pt_tmp = pd.read_excel("JSP_dataset.xlsx", sheet_name="Processing Time", index_col=[0])
ms_tmp = pd.read_excel("JSP_dataset.xlsx", sheet_name="Machines Sequence", index_col=[0])

# 机器数量
num_mc = pt_tmp.shape[1]
# 作业数量
num_job = pt_tmp.shape[0]
# 基因长度
num_gene = num_mc * num_job

# 加工时间
pt = [list(map(int, pt_tmp.iloc[i])) for i in range(num_job)]
# 机器顺序
ms = [list(map(int, ms_tmp.iloc[i])) for i in range(num_job)]

# print('pt')
# print(pt)
# print('ms')
# print(ms)

# 基因数量
population_size = 30
# 交叉率
crossover_rate = 0.8
# 变异率
mutation_rate = 0.2
# 变异选择率
mutation_selection_rate = 0.2
# 变异基因数
num_mutation_jobs = round(num_gene * mutation_selection_rate)
# 训练次数
num_iteration = 200

start_time = time.time()

'''----- generate initial population -----'''
Tbest = 999999999999999
best_list, best_obj = [], []
population_list = []
makespan_record = []
for i in range(population_size):
    nxm_random_num = list(np.random.permutation(num_gene))
    population_list.append(nxm_random_num)
    for j in range(num_gene):
        population_list[i][j] = population_list[i][j] % num_job

for n in range(num_iteration):
    Tbest_now = 99999999999

    '''-------- 交叉&繁殖 --------'''
    parent_list = copy.deepcopy(population_list)
    offspring_list = copy.deepcopy(population_list)
    S = list(np.random.permutation(population_size))

    for m in range(int(population_size / 2)):
        crossover_prob = np.random.rand()
        if crossover_rate >= crossover_prob:
            parent_1 = population_list[S[2 * m]][:]
            parent_2 = population_list[S[2 * m + 1]][:]
            child_1 = parent_1[:]
            child_2 = parent_2[:]
            cutpoint = list(np.random.choice(num_gene, 2, replace=False))
            cutpoint.sort()

            child_1[cutpoint[0]:cutpoint[1]] = parent_2[cutpoint[0]:cutpoint[1]]
            child_2[cutpoint[0]:cutpoint[1]] = parent_1[cutpoint[0]:cutpoint[1]]
            offspring_list[S[2 * m]] = child_1[:]
            offspring_list[S[2 * m + 1]] = child_2[:]

    '''----------修复基因-------------'''
    for m in range(population_size):
        job_count = {}
        larger, less = [], []
        for i in range(num_job):
            if i in offspring_list[m]:
                count = offspring_list[m].count(i)
                pos = offspring_list[m].index(i)
                job_count[i] = [count, pos]
            else:
                count = 0
                job_count[i] = [count, 0]
            if count > num_mc:
                larger.append(i)
            elif count < num_mc:
                less.append(i)

        for k in range(len(larger)):
            chg_job = larger[k]
            while job_count[chg_job][0] > num_mc:
                for d in range(len(less)):
                    if job_count[less[d]][0] < num_mc:
                        offspring_list[m][job_count[chg_job][1]] = less[d]
                        job_count[chg_job][1] = offspring_list[m].index(chg_job)
                        job_count[chg_job][0] = job_count[chg_job][0] - 1
                        job_count[less[d]][0] = job_count[less[d]][0] + 1
                    if job_count[chg_job][0] == num_mc:
                        break

    '''--------变异（循环）--------'''
    for m in range(len(offspring_list)):
        mutation_prob = np.random.rand()
        if mutation_rate >= mutation_prob:
            m_chg = list(
                np.random.choice(num_gene, num_mutation_jobs, replace=False))
            t_value_last = offspring_list[m][m_chg[0]]
            for i in range(num_mutation_jobs - 1):
                offspring_list[m][m_chg[i]] = offspring_list[m][m_chg[i + 1]]

            offspring_list[m][m_chg[
                num_mutation_jobs - 1]] = t_value_last

    '''fitness'''
    total_chromosome = copy.deepcopy(parent_list) + copy.deepcopy(offspring_list)
    chrom_fitness, chrom_fit = [], []
    total_fitness = 0
    for m in range(population_size * 2):
        # print("---------------------------------------------------------")
        j_keys = [j for j in range(num_job)]
        # print('j_keys->' + str(j_keys))
        key_count = {key: 0 for key in j_keys}
        # print('key_count->' + str(key_count))
        j_count = {key: 0 for key in j_keys}
        # print('j_count->' + str(j_count))
        m_keys = [j + 1 for j in range(num_mc)]
        # print('m_keys->' + str(m_keys))
        m_count = {key: 0 for key in m_keys}
        # print('m_count->' + str(m_count))
        # print("---------------------------------------------------------")
        # print()

        for i in total_chromosome[m]:
            # 时间
            gen_t = int(pt[i][key_count[i]])
            # 机器
            gen_m = int(ms[i][key_count[i]])
            j_count[i] = j_count[i] + gen_t
            m_count[gen_m] = m_count[gen_m] + gen_t

            if m_count[gen_m] < j_count[i]:
                m_count[gen_m] = j_count[i]
            elif m_count[gen_m] > j_count[i]:
                j_count[i] = m_count[gen_m]

            key_count[i] = key_count[i] + 1

        makespan = max(j_count.values())
        # chrom_fitness.append(1 / makespan)
        chrom_fitness.append(1 / (math.exp(0.2 * makespan)))
        chrom_fit.append(makespan)
        total_fitness = total_fitness + chrom_fitness[m]

    '''轮转筛选'''
    pk, qk = [], []

    for i in range(population_size * 2):
        pk.append(chrom_fitness[i] / total_fitness)
    for i in range(population_size * 2):
        cumulative = 0
        for j in range(0, i + 1):
            cumulative = cumulative + pk[j]
        qk.append(cumulative)

    selection_rand = [np.random.rand() for i in range(population_size)]

    for i in range(population_size):
        if selection_rand[i] <= qk[0]:
            population_list[i] = copy.deepcopy(total_chromosome[0])
        else:
            for j in range(0, population_size * 2 - 1):
                if selection_rand[i] > qk[j] and selection_rand[i] <= qk[j + 1]:
                    population_list[i] = copy.deepcopy(total_chromosome[j + 1])
                    break
    '''比较'''
    for i in range(population_size * 2):
        if chrom_fit[i] < Tbest_now:
            Tbest_now = chrom_fit[i]
            sequence_now = copy.deepcopy(total_chromosome[i])
    if Tbest_now <= Tbest:
        Tbest = Tbest_now
        sequence_best = copy.deepcopy(sequence_now)

    makespan_record.append(Tbest)

print("optimal result", sequence_best)
print("optimal value:%f" % Tbest)
print('time:%s' % (time.time() - start_time))

'''plotly'''
import pandas as pd
import plotly.offline as of
import plotly.graph_objs as go
import plotly.figure_factory as ff
import datetime

m_keys = [j + 1 for j in range(num_mc)]
j_keys = [j for j in range(num_job)]
key_count = {key: 0 for key in j_keys}
j_count = {key: 0 for key in j_keys}
m_count = {key: 0 for key in m_keys}
j_record = {}
for i in sequence_best:
    gen_t = int(pt[i][key_count[i]])
    gen_m = int(ms[i][key_count[i]])
    j_count[i] = j_count[i] + gen_t
    m_count[gen_m] = m_count[gen_m] + gen_t

    if m_count[gen_m] < j_count[i]:
        m_count[gen_m] = j_count[i]
    elif m_count[gen_m] > j_count[i]:
        j_count[i] = m_count[gen_m]

    start_time = str(datetime.timedelta(seconds=j_count[i] - pt[i][key_count[i]]))
    end_time = str(datetime.timedelta(seconds=j_count[i]))

    j_record[(i, gen_m)] = [start_time, end_time]

    key_count[i] = key_count[i] + 1

df = []
for m in m_keys:
    for j in j_keys:
        df.append(dict(Task='Machine %s' % (m),
                       Start='2020-01-01 %s' % (str(j_record[(j, m)][0])),
                       Finish='2020-01-01 %s' % (str(j_record[(j, m)][1])),
                       Resource='Job %s' % (j + 1)))

fig = ff.create_gantt(df, index_col='Resource', show_colorbar=True, group_tasks=True, showgrid_x=True,
                      title='Job shop Schedule')
of.offline.iplot(fig)
