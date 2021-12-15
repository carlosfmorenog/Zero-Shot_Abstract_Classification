def Kmeans(data, target, k):
    print('\nClass decomposition by applying k-means...')    
    from sklearn.cluster import KMeans
    target_clust = ['']*len(target)
    IndexesUnique = list(set(target))
    IndexesUnique.sort()
    for i, label in enumerate(IndexesUnique):
        if k[i]==1:
            print('No clustering performed for class '+str(label)+'.')
        else:
            print('Number of clusters for class '+str(label)+': '+str(k[i]))
        ## Split the dataset
        data_tocluster = []
        data_tocluster_index = []
        for j, dat in enumerate(data):
            if target[j]==label:
                data_tocluster.append(dat)
                data_tocluster_index.append(j)
        if 1<k[i]<=len(data_tocluster):
                ## Apply k-means to the list    
                kmeans = KMeans(n_clusters=k[i], random_state=0).fit(data_tocluster)
                for n, m in enumerate(kmeans.labels_):
                    target_clust[data_tocluster_index[n]]=str(label)+'_c'+str(m)
        else:
            for m in data_tocluster_index:
                target_clust[m]=str(label)+'_c0'
    return target_clust


def FCmeans(data, target, k):
    print('\nClass decomposition by applying FCmeans...')
    import skfuzzy as fuzz
    import numpy as np
    target_clust = ['']*len(target)
    IndexesUnique = list(set(target))
    IndexesUnique.sort()
    for i, label in enumerate(IndexesUnique):
        if k[i]==1:
            print('No clustering performed for class '+str(label)+'.')
        else:
            print('Number of clusters for class '+str(label)+': '+str(k[i]))
        ## Split the dataset
        data_tocluster = []
        data_tocluster_index = []
        for j, dat in enumerate(data):
            if target[j]==label:
                data_tocluster.append(dat)
                data_tocluster_index.append(j)
        data_tocluster_transpose = np.asarray(data_tocluster).transpose()
        if k[i]!=1 and len(data_tocluster)>1:
        ## Apply fcmeans
            _, u, _, _, _, _, fpc = fuzz.cluster.cmeans(data_tocluster_transpose, k[i], 2, error=0.005, maxiter=1000, init=None)
            cluster_membership = np.argmax(u, axis=0)
            for n, m in enumerate(cluster_membership):
                target_clust[data_tocluster_index[n]]=str(label)+'_c'+str(m)
        else:
            for m in data_tocluster_index:
                target_clust[m]=str(label)+'_c0'
    return target_clust

def FCmeansOptimised(data, target, max_nclusters = 10):
    print('\nClass decomposition by applying FCmeans Optimised...')
    import skfuzzy as fuzz
    import numpy as np
    target_clust = ['']*len(target)
    IndexesUnique = list(set(target))
    IndexesUnique.sort()
    for i, label in enumerate(IndexesUnique):
        fpcs = []
        us = []
        ## Split the dataset
        data_tocluster = []
        data_tocluster_index = []
        for j, dat in enumerate(data):
            if target[j]==label:
                data_tocluster.append(dat)
                data_tocluster_index.append(j)
        data_tocluster_transpose = np.asarray(data_tocluster).transpose()
        if len(data_tocluster)>1:
        ## Apply fcmeans for 2 to max_nclusters
            for ncenters in range(2,max_nclusters):
                if ncenters<=len(data_tocluster):
                    _, u, _, _, _, _, fpc = fuzz.cluster.cmeans(data_tocluster_transpose, ncenters, 2, error=0.005, maxiter=1000, init=None)
                    fpcs.append(fpc)
                    us.append(u)
            ## Find the best clustering (maximum fpc value)
            best_ncenters = fpcs.index(max(fpcs))
            print('Maximum fpc score for class '+str(label)+': '+str("%.2f" % (max(fpcs))+' ('+str(best_ncenters+2)+' clusters)'))
            cluster_membership = np.argmax(us[best_ncenters], axis=0)        
            for n, m in enumerate(cluster_membership):
                target_clust[data_tocluster_index[n]]=str(label)+'_c'+str(m)
        else:
            print('No clustering performed for class '+str(label)+'.')
            for m in data_tocluster_index:
                target_clust[m]=str(label)+'_c0'
    return target_clust


def DBSCAN(data, target, eps=0.5, min_samples=5, metric='cosine'):
    print('\nClass decomposition by applying DBSCAN...')
    from sklearn.cluster import DBSCAN
    target_clust = ['']*len(target)
    IndexesUnique = list(set(target))
    IndexesUnique.sort()
    for i, label in enumerate(IndexesUnique):
        ## Split the dataset
        data_tocluster = []
        data_tocluster_index = []
        for j, dat in enumerate(data):
            if target[j]==label:
                data_tocluster.append(dat)
                data_tocluster_index.append(j)
        if len(data_tocluster)>1:
        ## Apply DBSCAN
            db = DBSCAN(eps, min_samples, metric=metric).fit(data_tocluster)
            cluster_membership = db.labels_
            n_clusters = len(set(cluster_membership))
            min_samples+=1
            print('Number of clusters found for class '+str(label)+': '+str(n_clusters))
            for n, m in enumerate(cluster_membership):
                target_clust[data_tocluster_index[n]]=str(label)+'_c'+str(m)
        else:
            print('No clustering performed for class '+str(label)+'.')
            for m in data_tocluster_index:
                target_clust[m]=str(label)+'_c0'
    return target_clust