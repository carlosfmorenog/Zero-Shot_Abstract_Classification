#%% Code of the CDSMOTE method modified to work with multiclass datasets
# This variant of the method pslits the dataset into train and test for classification

# Please reference the following papers:
    
# Elyan E., Moreno-García C.F., Jayne C., CDSMOTE: class decomposition and synthetic minority class oversampling technique for imbalanced-data classification. Neural Comput Appl. 2020. doi:10.1007/s00521-020-05130-z
# @article{Elyan2020,
# author = {Elyan, Eyad and Moreno-Garcia, Carlos Francisco and Jayne, Chrisina},
# doi = {10.1007/s00521-020-05130-z},
# isbn = {0123456789},
# issn = {1433-3058},
# journal = {Neural Computing and Applications},
# publisher = {Springer London},
# title = {{CDSMOTE: class decomposition and synthetic minority class oversampling technique for imbalanced-data classification}},
# url = {https://doi.org/10.1007/s00521-020-05130-z},
# year = {2020}
# }

# IJCNN 2021


#%% 0. Import necessary packages

import sys
import clustData
import computeKVs
#import skmeans
import numpy as np
import csv
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

#%% ############################# 1. Input params #############################

database = 'AcevesMartins' # name of the folder where the csv file is
dataset = 'acevesmartins-2021Doc2vec' # name of the csv file containing the data and target
classdecomp = 'DBSCAN' # 'kmeans', 'FCmeans', 'FCmeansOptimised' and 'DBSCAN' available (SKMEANS coming soon)
metric_dbscan = 'cosine' # distance used to calculate the clustering in dbscan
                         # more options here: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html#sklearn.metrics.pairwise_distances
oversampler = 'SMOTE' # 'SMOTE' and 'ADASYN' available
n_clusters = 4 # used in option "fixed" and only if clusterings == "Kmeans" || "FCmeans"
classifier = 'SVM' # RF also available (MLP/NN/CNN coming soon)
number_of_tests = 5 # How many times to repeat the classification experiment comparing the original and new db
EPS = 0.5
threshold = 10 # if samples in positive class are apart from average by more than this value, apply oversampling (Sec 3.2 paper)
k_type = 'fixed' # Indicates how to calculate k values for class decomposition
# Choose from:
# "fixed": The majority class is decomposed using k=n_clusters
# "ir": The majority class is decomposed using k=ceil(IR), where IR is the imbalance ratio between the majority and the minority class. THIS RARELY LEADS TO OVERSAMPLING BEING USED!


#%% Loop

k_type = 'fixed'
metric_dbscan = 'cosine'
threshold = 10
oversampler = 'SMOTE'

list_datasets = [('AcevesMartins','acevesmartins_2021Doc2vec'),
                  ('AcevesMartins','acevesmartins_2021Fasttext'),
                  ('AcevesMartins','acevesmartins_2021Glo'),
                  ('AtypicalAntisychotics','AtypicalAntisychoticsDoc2vec'),
                  ('AtypicalAntisychotics','AtypicalAntisychoticsFasttext'),
                  ('AtypicalAntisychotics','AtypicalAntisychoticsGlo'),
                  ('BannachBrown','bannachbrown-2019doc2vec'),
                  ('BannachBrown','bannachbrown-2019fasttext'),
                  ('BannachBrown','bannachbrown-2019glo'),
                 ('Bos','bos_2018doc2vec'),
                 ('Bos','bos_2018fasttext'),
                 ('Bos','bos_2018glo'),
                 ('CohenCalcium','CalciumChannelBlockersDoc2vec'),
                 ('CohenCalcium','CalciumChannelBlockersFasttext'),
                 ('CohenCalcium','CalciumChannelBlockersGlo'),
                 ('CohenOral','OralHypoglycemicsDoc2vec'),
                 ('CohenOral','OralHypoglycemicsFasttext'),
                 ('CohenOral','OralHypoglycemicsGlo'),
                 ('VanDis','vandis2020doc2vec'),
                 ('VanDis','vandis2020fasttext'),
                 ('VanDis','vandis2020glo'),]
list_params = [(2,'kmeans',0),
               (3,'kmeans',0),
               (4,'kmeans',0),
               (5,'kmeans',0),
               (0,'DBSCAN',0.01),
               (0,'DBSCAN',0.1)]
list_classifs = ['SVM','RF']

for repo in list_datasets:
     for classif in list_classifs:    
         for params in list_params:        
            database = repo[0]
            dataset = repo[1]
            classdecomp = params[1]
            n_clusters = params[0]
            eps_DBSCAN = params[2]


            #%% ############################## 2. Load data ###############################
            
            ## 1. Load dataset
            with open('data//'+str(database)+'//'+str(dataset)+'.csv', 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                data = []
                target = []
                for row in reader:
                    data.append(list(map(float,row[0:len(row)-1])))
                    target.append(row[-1])
            del row, reader, f
            
            ## 1.a. 0-1 Standarisation, not used since datasets are very big and crashes
            # from sklearn import preprocessing
            # data = preprocessing.scale(data)
            
            ## 1.b. Correct targets
            for i,j in enumerate(target):
                if j == '0.000000000000000000e+00': 
                    target[i] = '0'
                elif j == '1.000000000000000000e+00': 
                    target[i] = '1'
                elif j == '2.000000000000000000e+00': # 2 is incorrect, should change to 0 (excluded)
                    target[i] = '0'
            
            ## 2. Find majority and minority classes
            majority_class = max(set(target), key=target.count)
            minority_class = min(set(target), key=target.count)
            
            ## 3. Plot distribution of original dataset
            print('\n###### Dataset: '+str(dataset)+' ##########\n')
            histo = [['Class','Number of Samples']]
            for i, label1 in enumerate(sorted(list(set(target)))):
                cont = 0
                for j, label2 in enumerate(target):
                    if label1 == label2:
                        cont+=1
                histo.append([label1,cont])
            histo.append(['Total Samples', len(target)])
            # Save the histogram as a .csv file   
            with open('results//'+database+'//originaldb_classdistribution_'+dataset+'_'+classdecomp+'_nclusters='+str(n_clusters)+'.csv', 'w', newline='', encoding='utf-8') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',')
                for i, hist in enumerate(histo):
                    filewriter.writerow(hist)
            # Load as a panda
            histo_panda = pd.DataFrame.from_records(histo[1:-1], columns=histo[0])
            print(histo_panda)
            print('Total samples: '+str(len(target)))
            # Create a histogram using seaborn
            sns_plot = sns.barplot(x="Class", y="Number of Samples", data=histo_panda)
            # Save the image
            sns_plot.figure.set_size_inches(10, 6)
            sns_plot.figure.savefig('results//'+database+'//originaldb_barchart_'+dataset+'_'+classdecomp+'_nclusters='+str(n_clusters)+'.jpg', orientation = 'landscape', dpi = 600)
            print('\nShowing class distribution bar chart...')
            plt.show()
                
            #%% ######################### 3. Class decomposition #########################
            
            ## 1. Calculate k vector (for class decomposition)
            
            if k_type.lower() == 'fixed':
                k = computeKVs.majority(data, target, n_clusters)
            elif k_type.lower() == 'ir':
                ## Calculate the IR between the majority and the minority
                majority_samples = histo_panda.loc[histo_panda['Class'] == majority_class].reset_index()
                minority_samples = histo_panda.loc[histo_panda['Class'] == minority_class].reset_index()
                n_clusters = math.ceil(majority_samples['Number of Samples'][0]/minority_samples['Number of Samples'][0])                                                                                                                                                                   
                k = computeKVs.majority(data, target, n_clusters)
            else:
                print('Invalid k values selecting option for CDSMOTE')
                sys.exit()
            
            ## 2. Cluster the data
            if classdecomp.lower()=='kmeans':
                target_cd = clustData.Kmeans(data, target, k)
            elif classdecomp.lower()=='fcmeans':
                target_cd = clustData.FCmeans(data, target, k)
            elif classdecomp.lower()=='fcmeansoptimised':
                 target_cd = clustData.FCmeansOptimised(data, target, max_nclusters = 10)   
            elif classdecomp.lower()=='dbscan':        
                 target_cd = clustData.DBSCAN(data, target, eps=EPS, metric=metric_dbscan)
#            elif classdecomp.lower()=='skmeans':
                
            else:
                print('Invalid clustering algorithm selected.')
                sys.exit()
                
            ## 3. Plot distribution after cd
            histo = [['Class','Number of Samples']]
            for i, label1 in enumerate(sorted(list(set(target_cd)))):
                cont = 0
                for j, label2 in enumerate(target_cd):
                    if label1 == label2:
                        cont+=1
                histo.append([label1,cont])
            histo.append(['Total Samples', len(target_cd)])
            # Save the histogram as a .csv file
            if classdecomp.lower()=='kmeans':
                nameoffile = 'results//'+database+'//decomposeddb_classdistribution_'+dataset+'_'+classdecomp+'_nclusters='+str(n_clusters)+'.csv'
            elif classdecomp.lower()=='dbscan':
                nameoffile = 'results//'+database+'//decomposeddb_classdistribution_'+dataset+'_'+classdecomp+'_eps='+str(eps_DBSCAN)+'.csv'    
            with open(nameoffile, 'w', newline='', encoding='utf-8') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',')
                for i, hist in enumerate(histo):
                    filewriter.writerow(hist)
            # Load as a panda
            histo_panda = pd.DataFrame.from_records(histo[1:-1], columns=histo[0])
            print(histo_panda)
            print('Total samples: '+str(len(target_cd)))
            # Create a histogram using seaborn
            sns_plot = sns.barplot(x="Class", y="Number of Samples", data=histo_panda)
            # draw a line depicting the average
            indexesUnique = list(set(target_cd))
            indexesUnique.sort()
            cdclassdist_count = []
            cdclassdist_names = []
            for cdclass in indexesUnique:
                 cdclassdist_count.append(target_cd.count(cdclass))
                 cdclassdist_names.append(cdclass)
            average = sum(cdclassdist_count)/len(cdclassdist_count)
            print('Average number of samples per class:', average)
            plt.axhline(average, color='red')
            # Save the image
            sns_plot.figure.set_size_inches(10, 6)
            if classdecomp.lower()=='kmeans':
                nameoffile = 'results//'+database+'//decomposeddb_barchart_'+dataset+'_'+classdecomp+'_nclusters='+str(n_clusters)+'.jpg'
            elif classdecomp.lower()=='dbscan':
                nameoffile = 'results//'+database+'//decomposeddb_barchart_'+dataset+'_'+classdecomp+'_eps='+str(eps_DBSCAN)+'.jpg' 
            sns_plot.figure.savefig(nameoffile, orientation = 'landscape', dpi = 600)
            print('\nShowing class distribution bar chart...')
            plt.show()
                 
            #%% ############################ 4. Oversampling #############################
            
            
            ## 1. Calculate reference class (i.e. closest to the average and above it) for oversampling
            c = np.inf
            ref = majority_class+'_c0' # gets picked by default if none other accomplishes
            for i,j in enumerate(cdclassdist_count):
                if abs(j-average)<c and j-average>=0:
                    c = abs(j-average)
                    ref = cdclassdist_names[i]
            
            data_cdsmote = list(np.asarray(data)[(np.where(np.asarray(target)==majority_class))])
            target_cdsmote = list(np.asarray(target_cd)[(np.where(np.asarray(target)==majority_class))])
            
            ## 2. For all non-majority classes (considering the original dataset), see if they are far (i.e. difference greater than the threshold) from the average (red line in the last plot)
            flag = 0
            for i, cdclassdist_name in enumerate(cdclassdist_names):
                if majority_class not in cdclassdist_name.split('_')[0]:
                    if abs(average-cdclassdist_count[i])>threshold and average-cdclassdist_count[i]>=0 and cdclassdist_count[i]>5: #>5 because smote can only oversample with over 5 samples!
                        flag = 1
                        print('Oversampling class '+str(cdclassdist_name)+'...')            
                        ## 3. Create a sub-dataset that only contains the new majority and current non-minority classes
                        data_majmin = []
                        target_majmin = []
                        for j, label in enumerate(target_cd):
                            if label == cdclassdist_name or label == ref:
                                data_majmin.append(data[j])
                                target_majmin.append(label)
                        ## 4. Do the oversampling
                        if oversampler.lower() == 'smote':
                            sm = SMOTE()
                            data_over, target_over = sm.fit_resample(data_majmin, target_majmin) 
                        elif oversampler.lower() == 'adasyn':
                            ada = ADASYN()
                            data_over, target_over = ada.fit_resample(data_majmin, target_majmin)
                        else:
                            print('Invalid oversampling algorithm.')
                            sys.exit() 
                        # Append the oversampled data to the new repository
                        for j, label in enumerate(target_over):
                            if label == cdclassdist_name:
                                data_cdsmote.append(list(data_over[j]))
                                target_cdsmote.append(label)
                    else:
                        # Append the not-oversampled
                        for j, label in enumerate(target_cd):
                            if label == cdclassdist_name:
                                data_cdsmote.append(list(data[j]))
                                target_cdsmote.append(label)
                                
            ## 5. If there is only one sample of a certain class, duplicate the entry
            for label in set(target_cdsmote):
                if target_cdsmote.count(label)==1:
                  index = target_cdsmote.index(label)
                  data_cdsmote.append(data_cdsmote[index])
                  target_cdsmote.append(target_cdsmote[index])
                        
                
            ## 6. Plot distribution after smote
            if flag == 1:
                histo = [['Class','Number of Samples']]
                for i, label1 in enumerate(sorted(list(set(target_cdsmote)))):
                    cont = 0
                    for j, label2 in enumerate(target_cdsmote):
                        if label1 == label2:
                            cont+=1
                    histo.append([label1,cont])
                histo.append(['Total Samples', len(target_cdsmote)])
                ## Save the histogram as a .csv file   
                if classdecomp.lower()=='kmeans':
                    nameoffile = 'results//'+database+'//cdsmotedb_classdistribution_'+dataset+'_'+classdecomp+'_nclusters='+str(n_clusters)+'.csv'
                elif classdecomp.lower()=='dbscan':
                    nameoffile = 'results//'+database+'//cdsmotedb_classdistribution_'+dataset+'_'+classdecomp+'_eps='+str(eps_DBSCAN)+'.csv'
                with open(nameoffile, 'w', newline='', encoding='utf-8') as csvfile:
                    filewriter = csv.writer(csvfile, delimiter=',')
                    for i, hist in enumerate(histo):
                        filewriter.writerow(hist)
                ## Load as a panda
                histo_panda = pd.DataFrame.from_records(histo[1:-1], columns=histo[0])
                print(histo_panda)
                print('Total samples: '+str(len(target_cdsmote)))
                ## Create a histogram using seaborn
                sns_plot = sns.barplot(x="Class", y="Number of Samples", data=histo_panda)
                ## draw a line depicting the average
                indexesUnique = list(set(target_cdsmote))
                indexesUnique.sort()
                newestclassdist_count = []
                for newestclass in indexesUnique:
                      newestclassdist_count.append(target_cdsmote.count(newestclass))
                average_new = sum(newestclassdist_count)/len(newestclassdist_count)
                print('New average number of samples per class:', average_new)
                plt.axhline(average, color='red')
                plt.axhline(average_new, color='blue')
                ## Save the image
                sns_plot.figure.set_size_inches(10, 6)
                if classdecomp.lower()=='kmeans':
                    nameoffile = 'results//'+database+'//cdsmotedb_barchart_'+dataset+'_'+classdecomp+'_nclusters='+str(n_clusters)+'.jpg'
                elif classdecomp.lower()=='dbscan':
                    nameoffile = 'results//'+database+'//cdsmotedb_barchart_'+dataset+'_'+classdecomp+'_eps='+str(eps_DBSCAN)+'.jpg'
                sns_plot.figure.savefig(nameoffile, orientation = 'landscape', dpi = 600)
                print('\nShowing class distribution bar chart...')
                plt.show()
            else:
                print('All non-majority classes are close to average. No oversampling was needed.')
            
            
            #%% Save the new dataset
            if classdecomp.lower()=='kmeans':
                nameoffile = 'data//'+str(database)+'//'+dataset+'_cdsmotedb_'+classdecomp+'_nclusters='+str(n_clusters)+'.csv'
            elif classdecomp.lower()=='dbscan':
                nameoffile = 'data//'+str(database)+'//'+dataset+'_cdsmotedb_'+classdecomp+'_eps='+str(eps_DBSCAN)+'.csv'
            with open(nameoffile, 'w', newline='') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',')
                for j, tar in enumerate(target_cdsmote):
                     row = list(data_cdsmote[j].copy())
                     row.append(target_cdsmote[j])
                     filewriter.writerow(row)
            
            
            #%% ########################### 5. Classification ############################
            
            print('\n')
            
            results = []
            accuracy_o_final = 0
            accuracy_c_final = 0
            recall_o_final = 0
            recall_c_final = 0
            precision_o_final = 0
            precision_c_final = 0
            f1_o_final = 0
            f1_c_final = 0
            

            ## Split data/target and data_cdsmote/target_cdsmote (stratified and with many splits)
            sss = StratifiedShuffleSplit(n_splits=number_of_tests, 
                                         test_size=0.3, random_state=42)
            
            # Original data, only do on the first test for the dataset to save time
            if n_clusters == 2:
                experiments = 0
                for train_index, test_index in sss.split(data, target):
                    # print(train_index, test_index)
                    print('\nExperiment '+str(experiments)+' Original DB...')
                    results.append(['Experiment '+str(experiments)+' Original DB...'])
                    experiments+=1
                    X_train_o, X_test_o = np.asarray(data)[train_index], np.asarray(data)[test_index]
                    y_train_o, y_test_o = np.asarray(target)[train_index], np.asarray(target)[test_index]
                    if classif.lower() == 'svm':
                        clf_o = svm.SVC(kernel='linear')
                    elif classif.lower() == 'rf':
                        clf_o = RandomForestClassifier(max_depth=2, random_state=0)
                    elif classif.lower() == 'nn':
                        clf_o = MLPClassifier(solver='adam', alpha=1e-5, max_iter=10000,
                                        hidden_layer_sizes=(5, 2), random_state=1)
                    clf_o.fit(X_train_o, y_train_o)
                    y_pred_o = clf_o.predict(X_test_o)
                    # Test         
                    print("Accuracy Original DB:",metrics.accuracy_score(y_test_o, y_pred_o))
                    results.append(["Accuracy Original DB:"+str(metrics.accuracy_score(y_test_o, y_pred_o))])
                    print("Precision Original DB:",metrics.precision_score(y_test_o, y_pred_o, average='weighted'))
                    results.append(["Precision Original DB:"+str(metrics.precision_score(y_test_o, y_pred_o, average='weighted'))])
                    print("Recall Original DB:",metrics.recall_score(y_test_o, y_pred_o, average='weighted'))
                    results.append(["Recall Original DB:"+str(metrics.recall_score(y_test_o, y_pred_o, average='weighted'))])
                    print("F1 Score Original DB:",metrics.f1_score(y_test_o, y_pred_o, average='weighted'))
                    results.append(["F1 Score Original DB:"+str(metrics.f1_score(y_test_o, y_pred_o, average='weighted'))])
                    
                    accuracy_o_final = accuracy_o_final + metrics.accuracy_score(y_test_o, y_pred_o)
                    precision_o_final = precision_o_final + metrics.precision_score(y_test_o, y_pred_o, average='weighted')
                    recall_o_final = recall_o_final + metrics.recall_score(y_test_o, y_pred_o, average='weighted')
                    f1_o_final = f1_o_final + metrics.f1_score(y_test_o, y_pred_o, average='weighted')
                    
                print('\n------')
                
            # CDSMOTE data
            experiments = 0
            for train_index, test_index in sss.split(data_cdsmote, target_cdsmote):
                print('\nExperiment '+str(experiments)+' CDSMOTE DB...')
                results.append(['Experiment '+str(experiments)+' CDSMOTE DB...'])
                experiments+=1
                X_train_c, X_test_c,  = np.asarray(data_cdsmote)[train_index], np.asarray(data_cdsmote)[test_index]
                y_train_c, y_test_c = np.asarray(target_cdsmote)[train_index], np.asarray(target_cdsmote)[test_index]
                if classif.lower() == 'svm':
                    clf_c = svm.SVC(kernel='linear')
                elif classif.lower() == 'rf':
                    clf_c = RandomForestClassifier(max_depth=2, random_state=0)
                elif classif.lower() == 'nn':
                    clf_c = MLPClassifier(solver='adam', alpha=1e-5, max_iter=10000,
                                    hidden_layer_sizes=(5, 2), random_state=1)
                clf_c.fit(X_train_c, y_train_c)
                y_pred_c = clf_c.predict(X_test_c)
                # Test, making sure accuracy considers sub_classes as good
                for i,label in enumerate(y_pred_c):
                      y_pred_c[i] = label.split('_')[0]
                for i,label in enumerate(y_test_c):
                      y_test_c[i] = label.split('_')[0]
                print("Accuracy CDSMOTE DB:",metrics.accuracy_score(y_test_c, y_pred_c))
                results.append(["Accuracy CDSMOTE DB:"+str(metrics.accuracy_score(y_test_c, y_pred_c))])
                print("Precision CDSMOTE DB:",metrics.precision_score(y_test_c, y_pred_c, average='weighted'))
                results.append(["Precision CDSMOTE DB:"+str(metrics.precision_score(y_test_c, y_pred_c, average='weighted'))])
                print("Recall CDSMOTE DB:",metrics.recall_score(y_test_c, y_pred_c, average='weighted'))
                results.append(["Recall CDSMOTE DB:"+str(metrics.recall_score(y_test_c, y_pred_c, average='weighted'))])
                print("F1 Score CDSMOTE DB:",metrics.f1_score(y_test_c, y_pred_c, average='weighted'))
                results.append(["F1 Score CDSMOTE DB:"+str(metrics.f1_score(y_test_c, y_pred_c, average='weighted'))])
                
                accuracy_c_final = accuracy_c_final + metrics.accuracy_score(y_test_c, y_pred_c)
                precision_c_final = precision_c_final + metrics.precision_score(y_test_c, y_pred_c, average='weighted')
                recall_c_final = recall_c_final + metrics.recall_score(y_test_c, y_pred_c, average='weighted')
                f1_c_final = f1_c_final + metrics.f1_score(y_test_c, y_pred_c, average='weighted')
            
            print('\n')
            
            # Final results
            print('Final Results:')
            results.append(['Final Results:'])
            if n_clusters == 2:
                print("\nAverage Accuracy Original DB:",accuracy_o_final/experiments)
                results.append(["Average Accuracy Original DB:"+str(accuracy_o_final/experiments)])
                print("Average Precision Original DB:",precision_o_final/experiments)
                results.append(["Average Precision Original DB:"+str(precision_o_final/experiments)])
                print("Average Recall Original DB:",recall_o_final/experiments)
                results.append(["Average Recall Original DB:"+str(recall_o_final/experiments)])
                print("Average F1 Score Original DB:",f1_o_final/experiments)
                results.append(["Average F1 Score Original DB:"+str(f1_o_final/experiments)])
            print("\nAverage Accuracy CDSMOTE DB:",accuracy_c_final/experiments)
            results.append(["Average Accuracy CDSMOTE DB:"+str(accuracy_c_final/experiments)])
            print("Average Precision CDSMOTE DB:",precision_c_final/experiments)
            results.append(["Average Precision CDSMOTE DB:"+str(precision_c_final/experiments)])
            print("Average Recall CDSMOTE DB:",recall_c_final/experiments)
            results.append(["Average Recall CDSMOTE DB:"+str(recall_c_final/experiments)])
            print("Average F1 Score CDSMOTE DB:",f1_c_final/experiments)
            results.append(["Average F1 Score CDSMOTE DB:"+str(f1_c_final/experiments)])
            
            #%%
            
            # Save results as a csv file
            if classdecomp.lower()=='kmeans':
                nameoffile = 'results//'+database+'//results_'+dataset+'_'+classdecomp+'_nclusters='+str(n_clusters)+'_'+classif+'.csv'
            elif classdecomp.lower()=='dbscan':
                nameoffile = 'results//'+database+'//results_'+dataset+'_'+classdecomp+'_eps='+str(eps_DBSCAN)+'_'+classif+'.csv'
            with open(nameoffile,'w', encoding='utf-8') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',')
                for i, res in enumerate(results):
                    filewriter.writerow(res)
