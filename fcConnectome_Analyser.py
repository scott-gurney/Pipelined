import os

from sklearn.svm import SVC

import NBS_vectorized_correlation
import numpy as np

from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, \
    classification_report, precision_recall_curve, average_precision_score, auc


from sklearn.ensemble import RandomForestClassifier
from NBS_vectorized_correlation import nbs_bct_corr_z
from nilearn import plotting, datasets
from numpy import genfromtxt
from nilearn.connectome import ConnectivityMeasure

import matplotlib.pyplot as plt

print("--------------------INITIALIZING ATLASES------------------------\n")
dataset = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr0-1mm")
#atlas_filename = dataset.maps
#labels = dataset.labels
atlas = datasets.fetch_atlas_msdl()
# Loading atlas image stored in 'maps'
atlas_filename = atlas["maps"]
# Loading atlas data stored in 'labels'
labels = atlas["labels"]
coords = atlas.region_coords

analysis = input("Please input 1 for NBS, and 2 for fc-MVPA.\n")

print("--------------------LOADING PATHS--------------------\n")
path_to_AD = (r'Unmasked\AD')
path_to_CN = (r'Unmasked\CN')
path_to_MCI = (r'Unmasked\MCI')
path_to_LMCI = (r'Unmasked\LMCI')
path_to_EMCI = (r'Unmasked\EMCI')


path_to_AD_mask = (r'GoodOutput\AD')
path_to_CN_mask = (r'GoodOutput\CN')
path_to_MCI_mask = (r'GoodOutput\MCI')
path_to_LMCI_mask = (r'GoodOutput\LMCI')
path_to_EMCI_mask = (r'GoodOutput\EMCI')
cohorts = ["CN", "EMCI", "MCI", "LMCI", "AD"]

def extractConnectome(data):
    correlation_measure = ConnectivityMeasure(kind="covariance")
    correlation_matrix = correlation_measure.fit_transform([data])[0]
    np.fill_diagonal(correlation_matrix, 0)
    return correlation_matrix


def extractTime(path):
    time = []
    connect = []
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if os.path.isfile(filepath):
            title = filename
            print(title)
            my_data = genfromtxt(filepath, delimiter=',')
            time.append(my_data)
            connect.append(extractConnectome(my_data))
    return time, connect


def combineCohorts(cohort1, cohort2):
    combined_array = []
    for i in range(len(cohort1)):
        combined_array.append(cohort1[i])
    for j in range(len(cohort2)):
        combined_array.append(cohort2[j])
    np_combined = np.array(combined_array)
    np_combined = np.transpose(np_combined, (1, 2, 0))
    arr = np.zeros(len(cohort1) + len(cohort2))
    arr[len(cohort1):] = 1
    return np_combined, arr

def findBest(combinedConnectome, vector):
    checker = 1
    r = 0.25
    sign = 1
    int1 = r
    int2 = r
    current = 0
    try:
        while checker > 0.05:
            pval, adj, null = nbs_bct_corr_z(combinedConnectome, r, vector)
            if sign == 1:
                int2 -= 0.02
                current = int2
            elif sign == -1:
                int1 += 0.02
                current = int1

            if pval is not None:
                if isinstance(pval, list):
                    for i in pval:
                        if i > 0.05:
                            sign *= -1
                            r = round(current, 2)
                            print(r)
                            continue
                        elif i < checker:
                            checker = i
                            r = round(current, 2)
                            print(r)
                        else:
                            r = round(current, 2)
                            print(r)
                            continue
                else:
                    if pval > 0.05:
                        sign *= -1
                        r = round(current, 2)
                        print(r)
                        continue
                    elif pval < checker:
                        checker = pval
                        r = round(current, 2)
                        print(r)
                    else:
                        r = round(current, 2)
                        print(r)
                        continue
            else:
                return ("Error pval is none")
        return r
    except ValueError:
        return r

def prepare_feature_matrix(correlation_arrays):
    flattened_matrices = []
    for array in correlation_arrays:
        for matrix in array:
            flattened_matrices.append(matrix.flatten())
    feature_matrix = np.vstack(flattened_matrices)

    return feature_matrix

def prepareTargetArray(correlation_arrays):
    targets = []
    for i, array in enumerate(correlation_arrays):
        targets.extend([i] * len(array))
    target_array = np.array(targets)
    return target_array

def prepareFilteredFeatureMatrix(correlation_arrays, nbs_matrix):
    flattened_matrices = []
    for array in correlation_arrays:
        filtered_matrix = array * nbs_matrix
        flattened_matrices.append(filtered_matrix.flatten())

    feature_matrix = np.vstack(flattened_matrices)

    return feature_matrix

def grangerCausality(cohort1, cohort2, max):
    results = []
    labelss = []
    counter1 = 0
    for each1 in cohort1:
        counter2 = 0
        counter1 += 1
        for each2 in cohort2:
            counter2 += 1
            for i in range(len(labels)):
                try:
                    comp = np.column_stack([each1[:, i], each2[:, i]])
                    result = grangercausalitytests(comp, maxlag=max)
                    for lag, data in result.items():
                        # Unpack the test results and models
                        test_results, models = data
                        # Iterate over the test results
                        for test, values in test_results.items():
                            # Unpack the test values
                            if len(values) == 4:
                                stat, pval, _, _ = values
                            elif len(values) == 3:
                                stat, pval, _, = values
                            else:
                                print("Cool")
                                # Check if the p-value is less than 0.05
                            if pval < 0.05:
                                labelss.append(labels[i])
                                results.append(result)
                            else:
                                continue
                except ValueError:
                    print(str(counter1) + str(counter2))
    return results, labelss
def doNBS():
    print("--------------------STARTING NBS--------------------\n")

    #UNCOMMENT TO DO NBS
    #bestr1 = findBest(cn_emci_connect, vec1)
    bestr1 = 0.3
    # pval, adj, null = nbs_bct_corr_z(cn_emci_connect, bestr1, vec1)
    # nilearn.plotting.plot_matrix(adj, labels=labels, colorbar=True)
    #
    # np.savetxt("cn_emci_adj.csv", adj, delimiter=",")
    # np.savetxt("cn_emci_pval.csv", pval, delimiter=",")
    # np.savetxt("cn_emci_null.csv", null, delimiter=",")
    #
    # #bestr2 = findBest(emci_mci_connect, vec2)
    # pval, adj, null = nbs_bct_corr_z(emci_mci_connect, 0.4, vec2)
    # nilearn.plotting.plot_matrix(adj, labels=labels, colorbar=True)
    #
    # np.savetxt("emci_mci_adj.csv", adj, delimiter=",")
    # np.savetxt("emci_mci_pval.csv", pval, delimiter=",")
    # np.savetxt("emci_mci_null.csv", null, delimiter=",")

    #bestr3 = findBest(mci_lmci_connect, vec3)
    # pval, adj, null = nbs_bct_corr_z(mci_lmci_connect, bestr1, vec3)
    # nilearn.plotting.plot_matrix(adj, labels=labels, colorbar=True)
    #
    # np.savetxt("mci_lmci_adj.csv", adj, delimiter=",")
    # np.savetxt("mci_lmci_pval.csv", pval, delimiter=",")
    # np.savetxt("mci_lmci_null.csv", null, delimiter=",")
    #
    # #bestr4 = findBest(emci_mci_connect, vec4)
    # pval, adj, null = nbs_bct_corr_z(lmci_ad_connect, bestr1, vec4)
    # nilearn.plotting.plot_matrix(adj, labels=labels, colorbar=True)
    #
    # np.savetxt("lmci_ad_adj.csv", adj, delimiter=",")
    # np.savetxt("lmci_ad_pval.csv", pval, delimiter=",")
    # np.savetxt("lmci_ad_null.csv", null, delimiter=",")


print("------------------------EXTRACTING CONNECTOMES--------------------\n")

# COMMENT THIS BLOCK OUT TO ONLY DO fc-MVPA
ad_time, ad_connect = extractTime(path_to_AD)
lmci_time, lmci_connect = extractTime(path_to_LMCI)
mci_time, mci_connect = extractTime(path_to_MCI)
emci_time, emci_connect = extractTime(path_to_EMCI)
cn_time, cn_connect = extractTime(path_to_CN)

# COMMENT THIS BLOCK OUT TO ONLY DO NBS
_, ad_connect_mask = extractTime(path_to_AD_mask)
_, lmci_connect_mask = extractTime(path_to_LMCI_mask)
_, mci_connect_mask = extractTime(path_to_MCI_mask)
_, emci_connect_mask = extractTime(path_to_EMCI_mask)
_, cn_connect_mask = extractTime(path_to_CN_mask)


print("--------------------COMBINING MATRICES--------------------\n")

# UNCOMMENT THIS TO START MATRIX COMBINATION, NEEDED FOR NBS

#cn_mci_connect, vec1 = combineCohorts(cn_big_matrix, mci_big_matrix)
#emci_mci_connect, vec2 = combineCohorts(emci_big_matrix, mci_big_matrix)
#cn_ad_connect, vec2 = combineCohorts(cn_big_matrix, ad_big_matrix)
#mci_lmci_connect, vec3 = combineCohorts(mci_big_matrix, lmci_big_matrix)
#lmci_ad_connect, vec4 = combineCohorts(lmci_big_matrix, ad_big_matrix)
#lmci_ad_connect, vec4 = combineCohorts(mci_big_matrix, ad_big_matrix)

# UNCOMMENT THIS TO DO NBS ON RESPECTIVE COHORT COMPARISONS
# bestr1 = 0.3
# pval, adj, null = nbs_bct_corr_z(cn_mci_connect, bestr1, vec1)
# nilearn.plotting.plot_matrix(adj, labels=labels, colorbar=True)
#
# np.savetxt("cn_mci_adj.csv", adj, delimiter=",")
# np.savetxt("cn_mci_pval.csv", pval, delimiter=",")
# np.savetxt("cn_mci_null.csv", null, delimiter=",")

# bestr1 = 0.4
# pval, adj, null = nbs_bct_corr_z(cn_ad_connect, bestr1, vec2)
# nilearn.plotting.plot_matrix(adj, labels=labels, colorbar=True)
#
# np.savetxt("cn_ad_adj.csv", adj, delimiter=",")
# np.savetxt("cn_ad_pval.csv", pval, delimiter=",")
# np.savetxt("cn_ad_null.csv", null, delimiter=",")
# for i in range(len(cohorts) - 1):
#     cohort1_name = cohorts[i]
#     cohort2_name = cohorts[i + 1]
#     print(f"Comparing cohort {cohort1_name} with {cohort2_name}")
#
#     results, labelsss = grangerCausality(cn_time, emci_time, 2)
#     for i in range(len(results)):
#         print(results[i])
#         print(labelsss[i])

#doNBS()

filter1 = np.loadtxt(r'NBS_Masks\cn_mci_adj.csv', delimiter=",", dtype=float)
filter2 = np.loadtxt(r'NBS_Masks\mci_lmci_adj.csv', delimiter=",", dtype=float)
filter3 = np.loadtxt(r'NBS_Masks\mci_ad_adj.csv', delimiter=",", dtype=float)
filter4 = np.loadtxt(r'NBS_Masks\cn_ad_adj.csv', delimiter=",", dtype=float)

# nilearn.plotting.plot_matrix(filter1, labels=labels)
# nilearn.plotting.show()
# nilearn.plotting.plot_matrix(filter2, labels=labels)
# nilearn.plotting.show()
filtered5 = prepareFilteredFeatureMatrix(cn_connect, filter1)
filtered4 = prepareFilteredFeatureMatrix(emci_connect, filter3)
filtered3 = prepareFilteredFeatureMatrix(mci_connect, filter2)
filtered2 = prepareFilteredFeatureMatrix(lmci_connect, filter3)
filtered1 = prepareFilteredFeatureMatrix(ad_connect, filter3)

#filtered6 = prepareFilteredFeatureMatrix(emci_test_connect, filter4)

if analysis == 1:
    feature_matrix = prepare_feature_matrix([filtered1, filtered2, filtered3, filtered4, filtered5])
else:
    feature_matrix = prepare_feature_matrix([ad_connect_mask, lmci_connect_mask, mci_connect_mask,
                                            emci_connect_mask, cn_connect_mask])
#filtered1, filtered2, filtered3, filtered4, filtered5
if analysis == 1:
    target_array = prepareTargetArray([filtered1, filtered2, filtered3, filtered4, filtered5])
else:
    target_array = prepareTargetArray([ad_connect_mask, lmci_connect_mask, mci_connect_mask,
                                            emci_connect_mask, cn_connect_mask])


#USE THIS TO TEST INDIVIDUAL COHORTS
# a = []
# b = []
# d = []
# e = []
# test_array = prepare_feature_matrix([a, filtered2, e, d])
# y_tester = prepareTargetArray([a, filtered2, e, d])



target_array = label_binarize(target_array, classes=[0, 1, 2, 3, 4])
n_classes = target_array.shape[1]
X_train, X_test, y_train, y_test = train_test_split(feature_matrix, target_array, test_size=.5, random_state=42)
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=1, min_samples_split=2, random_state=42),
    "SVM": svm.SVC(kernel='linear', C=10, gamma=0.01, random_state=42, probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=3, weights='uniform')
}

classes = ["AD", "LMCI", "MCI", "EMCI", "CN"]
for name, classifier in classifiers.items():
    classifier = OneVsRestClassifier(classifier)
    classifier.fit(X_train, y_train)
    y_score = classifier.predict_proba(X_test)
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        auc_score = auc(recall[i], precision[i])
        print("Class :" + str(n_classes) + " auc score = " + str(auc_score))
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    plt.figure()
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], lw=2, label=classes[i])

    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title(f"Precision vs. Recall curve ({name})")
    plt.show()

#
# #=======================================VARIOUS TEST HARNESSES======================================

#===================================TEST VARIOUS HYPERMARAMETERS=====================================
# classifiers = {
#     'K-Nearest Neighbors': KNeighborsClassifier(),
#     'SVM': SVC(),
#     'Random Forest': RandomForestClassifier()
# }
#
# param_grid = {
#     "Rotation Forest" : {
#         "n_trees" : [10, 50, 100, 200, 300],
#         "n_features" :[3],
#         "bootstrap" : [True],
#     },
#     "Random Forest": {
#         "n_estimators": [10, 50, 100, 200, 300],
#         "max_depth": [None, 5, 10, 20],
#         "min_samples_split": [2],
#         "min_samples_leaf": [1],
#     },
#     "SVM": {
#         "C": [0.001, 0.01, 0.1, 1, 10, 100],
#         "gamma": [1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
#     },
#     "K-Nearest Neighbors": {
#         "n_neighbors": [3, 5, 7, 9],
#         "weights": ['uniform', 'distance'],
#     },
# }
#
# skf = StratifiedKFold(n_splits=10)
#
# for name, clf in classifiers.items():
#     gs = GridSearchCV(clf, param_grid[name], cv=skf, n_jobs=-1)
#     gs.fit(feature_matrix, target_array)
#     print(f"Best parameters for {name}: {gs.best_params_}")
#     print(f"Best score for {name}: {gs.best_score_}")
# #
#
# ===============================================GET F1-SCORE======================================================
classifiers = {
    "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=200),
    "SVM": SVC(kernel='linear', C=1, gamma=0.1),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, weights='uniform')
}

skf = StratifiedKFold(n_splits=10)

for name, clf in classifiers.items():
    confusion_matrices = []
    average = []
    class_reports = []

    for train_index, test_index in skf.split(feature_matrix, target_array):
        X_train, X_test = feature_matrix[train_index], feature_matrix[test_index]
        y_train, y_test = target_array[train_index], target_array[test_index]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        average.append(accuracy)
        conf = confusion_matrix(y_test, y_pred, labels=np.unique(target_array))
        confusion_matrices.append(conf)
        # Get classification report for this fold
        report = classification_report(y_test, y_pred, output_dict=True)
        class_reports.append(report)
    avg_conf_matrix = np.mean(confusion_matrices, axis=0)
    print("Overall Accuracy :" + str((sum(average) / len(average))))

    classes = np.unique(target_array)
    avg_report = {class_: {'precision': 0, 'recall': 0, 'f1-score': 0} for class_ in classes}
    for report in class_reports:
        for class_ in classes:
            avg_report[class_]['precision'] += report[str(class_)]['precision']
            avg_report[class_]['recall'] += report[str(class_)]['recall']
            avg_report[class_]['f1-score'] += report[str(class_)]['f1-score']

    for class_ in classes:
        avg_report[class_]['precision'] /= len(class_reports)
        avg_report[class_]['recall'] /= len(class_reports)
        avg_report[class_]['f1-score'] /= len(class_reports)

    print("Average Precision, Recall, F1-score per class:", avg_report)

    disp = ConfusionMatrixDisplay(avg_conf_matrix, display_labels=["AD", "LMCI", "MCI", "EMCI", "CN"])
    disp.plot()
    plt.show()

# ======================================SHOW CONFUSION MATRICES============================================
for name, clf in classifiers.items():
    confusion_matrices = []
    average = []
    for train_index, test_index in skf.split(feature_matrix, target_array):
        X_train, X_test = feature_matrix[train_index], feature_matrix[test_index]
        y_train, y_test = target_array[train_index], target_array[test_index]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        average.append(accuracy)
        confusion_matrices.append(confusion_matrix(y_test, y_pred, labels=np.unique(target_array)))

    avg_confusion_matrix = np.mean(confusion_matrices, axis=0)
    print("Accuracy :" + str((sum(average)/len(average))))
    disp = ConfusionMatrixDisplay(avg_confusion_matrix, display_labels=["CN", "EMCI", "MCI", "LMCI", "AD"])
    disp.plot()
    plt.show()

