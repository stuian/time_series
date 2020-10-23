import numpy as np

def neigbor_matrix(part):
    N = len(part)
    neigborhood = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            if part[i] == part[j]:
                neigborhood[i,j] = 1
            else:
                neigborhood[i,j] = 0
            neigborhood[j,i] = neigborhood[i,j]
    return neigborhood

def two_level_consistency(parts,value): # 0.8
    N = parts.shape[1]
    must_link = np.zeros((N, N))
    cannot_link = np.zeros((N,N))
    for b in range(len(parts)):
        neigborhood = neigbor_matrix(parts[b])
        must_link += neigborhood
        neigborhood += 1
        neigborhood[neigborhood > 1] = 0
        cannot_link += neigborhood
    must_link[must_link >= value] = 1
    cannot_link[cannot_link >=value] = 1
    # 1、The clustering-level consistency
    u = []
    whole_constraints = (np.sum(must_link) + np.sum(cannot_link)) / 2
    for b in range(len(parts)):
        count = 0
        for i in range(N):
            for j in range(i + 1, N):
                if must_link[i,j] == 1:
                    if parts[b][i] == parts[b][j]:
                        count += 1
                if cannot_link[i,j] == 1:
                    if parts[b][i] != parts[b][j]:
                        count += 1

        u.append(count / whole_constraints)

    cluster_contribute = []
    for b in range(len(parts)):
        part = parts[b]
        K = len(np.unique(part))
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
            samples = in_cluster[k]
            cluster_length = len(samples)
            expect = (cluster_length / N) * whole_constraints
            count = 0
            whole = 0
            for i in range(cluster_length):
                for j in range(i + 1, cluster_length):
                    if must_link[samples[i], samples[j]] == 1:
                        whole += 1
                        if parts[b][samples[i]] == parts[b][samples[j]]:
                            count += 1
                    if cannot_link[samples[i], samples[j]] == 1:
                        whole += 1
                        if parts[b][samples[i]] != parts[b][samples[j]]:
                            count += 1
            cluster_contri = count / whole
            # cluster_weight = whole/expect  # ?
            cluster_contribute.append(cluster_contri*(cluster_contri + u[b]))
    return cluster_contribute