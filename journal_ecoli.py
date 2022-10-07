import random
import pandas
import numpy as np
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict
import lightgbm as lgb
#import imbalance_xgboost as imb_xgb
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import EditedNearestNeighbours

class CRO_class:

    def __init__(self):
        global NumHit, T_NumHit, MinStruct, MinPE, MinHit, buffer, KELossRate, MoleColl, alpha, beta
        global TotalMolecule, PE, KE, MoleNumber, molecule, copy_molecules, copy_molecules2
        global X, Y, y, features
        # Create Random Molecules
        MoleNumber = random.randint(1, 50)
        # Create A List of Possible Total Number of Molecules
        molecule = [[0 for i in range(184)] for j in range(MoleNumber)]
        copy_molecules = [[0 for i in range(184)] for j in range(MoleNumber)]
        copy_molecules2 = [[0 for i in range(184)] for j in range(MoleNumber)]
        MinStruct = ['SG','DG','SG_DG_SUM','SG_DG_PRO','SG_DG_SUB','EV','SG_EV_SUM','SG_EV_PRO','SG_EV_SUB','DG_EV_SUM','DG_EV_PRO','DG_EV_SUB','INFO','SG_INFO_SUM','SG_INFO_PRO','SG_INFO_SUB','DG_INFO_SUM',	'DG_INFO_PRO','DG_INFO_SUB','EV_INFO_SUM','EV_INFO_PRO','EV_INFO_SUB','LAC','SG_LAC_SUM','SG_LAC_PRO','SG_LAC_SUB','DG_LAC_SUM','DG_LAC_PRO','DG_LAC_SUB','EV_LAC_SUM','EV_LAC_PRO','EV_LAC_SUB','INFO_LAC_SUM','INFO_LAC_PRO','NFO_LAC_SUB','BNESS','SG_BNESS_SUM','SG_BNESS_PRO','SG_BNESS_SUB','DG_BNESS_SUM','DG_BNESS_PRO','DG_BNESS_SUB','EV_BNESS_SUM','EV_BNESS_PRO','EV_BNESS_SUB','INFO_BNESS_SUM','INFO_BNESS_PRO','INFO_BNESS_SUB','LAC_BNESS_SUM','LAC_BNESS_PRO','LAC_BNESS_SUB','CNESS','SG_CNESS_SUM','SG_CNESS_PRO','SG_CNESS_SUB','DG_CNESS_SUM','DG_CNESS_PRO','DG_CNESS_SUB','EV_CNESS_SUM','EV_CNESS_PRO','EV_CNESS_SUB','INFO_CNESS_SUM','INFO_CNESS_PRO','INFO_CNESS_SUB','LAC_CNESS_SUM','LAC_CNESS_PRO','LAC_CNESS_SUB','BNESS_CNESS_SUM','BNESS_CNESS_PRO','BNESS_CNESS_SUB','NET','SG_NET_SUM','SG_NET_PRO','SG_NET_SUB','DG_NET_SUM','DG_NET_PRO','DG_NET_SUB','EV_NET_SUM','EV_NET_PRO','EV_NET_SUB','INFO_NET_SUM','INFO_NET_PRO','INFO_NET_SUB','LAC_NET_SUM','LAC_NET_PRO','LAC_NET_SUB','BNESS_NET_SUM','BNESS_NET_PRO','BNESS_NET_SUB','CNESS_NET_SUM','CNESS_NET_PRO','CNESS_NET_SUB','GE_T1','GE_T1_GE_T2_SUM','GE_T1_GE_T2_PRO','GE_T1_GE_T2_SUB','GE_T1_GE_T3_SUM','GE_T1_GE_T3_PRO','GE_T1_GE_T3_SUB','GE_T1_GE_T4_SUM','GE_T1_GE_T4_PRO','GE_T1_GE_T4_SUB','GE_T1_GE_T5_SUM','GE_T1_GE_T5_PRO','GE_T1_GE_T5_SUB','GE_T1_GE_T6_SUM','GE_T1_GE_T6_PRO','GE_T1_GE_T6_SUB','GE_T1_GE_T7_SUM','GE_T1_GE_T7_PRO','GE_T1_GE_T7_SUB','GE_T1_GE_T8_SUM','GE_T1_GE_T8_PRO','GE_T1_GE_T8_SUB','GE_T2','GE_T2_GE_T3_SUM','GE_T2_GE_T3_PRO','GE_T2_GE_T3_SUB','GE_T2_GE_T4_SUM','GE_T2_GE_T4_PRO','GE_T2_GE_T4_SUB','GE_T2_GE_T5_SUM','GE_T2_GE_T5_PRO','GE_T2_GE_T5_SUB','GE_T2_GE_T6_SUM','GE_T2_GE_T6_PRO','GE_T2_GE_T6_SUB','GE_T2_GE_T7_SUM','GE_T2_GE_T7_PRO','GE_T2_GE_T7_SUB','GE_T2_GE_T8_SUM','GE_T2_GE_T8_PRO','GE_T2_GE_T8_SUB','GE_T3','GE_T3_GE_T4_SUM','GE_T3_GE_T4_PRO','GE_T3_GE_T4_SUB','GE_T3_GE_T5_SUM','GE_T3_GE_T5_PRO','GE_T3_GE_T5_SUB','GE_T3_GE_T6_SUM','GE_T3_GE_T6_PRO','GE_T3_GE_T6_SUB','GE_T3_GE_T7_SUM','GE_T3_GE_T7_PRO','GE_T3_GE_T7_SUB','GE_T3_GE_T8_SUM','GE_T3_GE_T8_PRO','GE_T3_GE_T8_SUB','GE_T4','GE_T4_GE_T5_SUM','GE_T4_GE_T5_PRO','GE_T4_GE_T5_SUB','GE_T4_GE_T6_SUM','GE_T4_GE_T6_PRO','GE_T4_GE_T6_SUB','GE_T4_GE_T7_SUM','GE_T4_GE_T7_PRO','GE_T4_GE_T7_SUB','GE_T4_GE_T8_SUM','GE_T4_GE_T8_PRO','GE_T4_GE_T8_SUB','GE_T5','GE_T5_GE_T6_SUM','GE_T5_GE_T6_PRO','GE_T5_GE_T6_SUB','GE_T5_GE_T7_SUM','GE_T5_GE_T7_PRO','GE_T5_GE_T7_SUB','GE_T6','GE_T6_GE_T8_SUM','GE_T6_GE_T8_PRO','GE_T6_GE_T8_SUB','GE_T7','GE_T7_GE_T8_SUM','GE_T7_GE_T8_PRO','GE_T7_GE_T8_SUB','GE_T8']
        MinPE = 0
        MinHit = 0
        buffer = 100
        KELossRate = .2
        MoleColl = .6
        alpha = 70
        beta = 30
        TotalMolecule = 0
        T_NumHit = 0
        # Assign Initial Activation Values to Molecules
        for row in range(MoleNumber):
            for j in range(184):
                molecule[row][j] = random.randint(0, 1)

        showtoaltmolecules(self)

def showtoaltmolecules(self):

    print("Initial Molecule Number: " + format(MoleNumber))
    delete_repeatmolecules(self)

def delete_repeatmolecules(self):
    # Create a Copy of Activation Values List
    global copy_molecules, copy_molecules2, molecule, TotalMolecule,deleted_molecules_number, MoleNumber
    copy_molecules2 = molecule.copy()
    copy_molecules.clear()
    pos = 0
    count = 0
    # Search molecules with repeat activation values
    for row in range(MoleNumber):
        found = 0
        for row2 in range(row + 1, MoleNumber):
            similar = 0
            all_zero = 0
            for j in range(184):
                if molecule[row][j] == copy_molecules2[row2][j]:
                    similar = similar + 1
                if molecule[row][j] == 0:
                    all_zero = all_zero + 1

            if similar >= 183 or all_zero >= 183:
               found = 1
               count = count + 1
               break
        if found == 1:
            continue
        copy_molecules.append(molecule[row])
        print(molecule[row])
        pos = pos + 1
    # Delete repeat molecules with same activation values
    print('Repeated and No feature Molecules: ' + format(count))

    molecule = [[]]
    molecule = copy_molecules.copy()
    TotalMolecule = len(molecule)
    print('Remaining Molecules: ' + format(TotalMolecule))

    for row in range(TotalMolecule):
        print(format(row)+'. '+format(molecule[row]))
    print("Molecule Number For CRO: " + format(TotalMolecule))

    read_data(self)

def read_data(self):
    global PE, KE, features, X, y, row, df, NumHit, molecule, copy_molecules, copy_molecules2, TotalMolecule
    global SG_X, DG_X, SG_DG_SUM_X, SG_DG_PRO_X, SG_DG_SUB_X, EV_X, SG_EV_SUM_X, SG_EV_PRO_X, SG_EV_SUB_X, DG_EV_SUM_X, DG_EV_PRO_X, DG_EV_SUB_X, INFO_X, SG_INFO_SUM_X, SG_INFO_PRO_X, SG_INFO_SUB_X, DG_INFO_SUM_X, DG_INFO_PRO_X, DG_INFO_SUB_X, X_smote, y_smote
    global EV_INFO_SUM_X, EV_INFO_PRO_X, EV_INFO_SUB_X, LAC_X, SG_LAC_SUM_X, SG_LAC_PRO_X, SG_LAC_SUB_X, DG_LAC_SUM_X, DG_LAC_PRO_X, DG_LAC_SUB_X, EV_BNESS_SUM_X, EV_BNESS_PRO_X, EV_BNESS_SUB_X, INFO_BNESS_SUM_X, INFO_BNESS_PRO_X, INFO_BNESS_SUB_X
    global EV_LAC_SUM_X, EV_LAC_PRO_X, EV_LAC_SUB_X, INFO_LAC_SUM_X, INFO_LAC_PRO_X, INFO_LAC_SUB_X, BNESS_X, SG_BNESS_SUM_X, SG_BNESS_PRO_X, SG_BNESS_SUB_X, DG_BNESS_SUM_X, DG_BNESS_PRO_X, DG_BNESS_SUB_X
    global INFO_BNESS_SUM_X, INFO_BNESS_PRO_X, INFO_BNESS_SUB_X, LAC_BNESS_SUM_X, LAC_BNESS_PRO_X, LAC_BNESS_SUB_X, CNESS_X, SG_CNESS_SUM_X, SG_CNESS_PRO_X, SG_CNESS_SUB_X, DG_CNESS_SUM_X, DG_CNESS_PRO_X, DG_CNESS_SUB_X, EV_CNESS_SUM_X, EV_CNESS_PRO_X, EV_CNESS_SUB_X
    global INFO_CNESS_SUM_X, INFO_CNESS_pro_X, INFO_CNESS_SUB_X, LAC_CNESS_SUM_X, LAC_CNESS_PRO_X, LAC_CNESS_SUB_X, BNESS_CNESS_SUM_X, BNESS_CNESS_PRO_X, BNESS_CNESS_SUB_X, NET_X, SG_NET_SUM_X, SG_NET_PRO_X, SG_NET_SUB_X, DG_NET_SUM_X, DG_NET_PRO_X, DG_NET_SUB_X, EV_NET_SUM_X, EV_NET_PRO_X, EV_NET_SUB_X, INFO_NET_SUM_X, INFO_NET_PRO_X, INFO_NET_SUB_X
    global LAC_NET_SUM_X, LAC_NET_PRO_X, LAC_NET_SUB_X, BNESS_NET_SUM_X, BNESS_NET_PRO_X, BNESS_NET_SUB_X, CNESS_NET_SUM_X, CNESS_NET_PRO_X, CNESS_NET_SUB_X, F_X

    PE = [TotalMolecule]
    KE = [TotalMolecule]
    NumHit = [TotalMolecule]
    features = [100]
    copy_molecules.clear()
    copy_molecules = molecule.copy()

    df = pandas.read_csv("E:\ecoli.csv", header=0)
    df.head()

    scaler = StandardScaler()

    SG_X = np.array(df["SG"]).reshape(-1, 1)
    df["SG"] = scaler.fit_transform(SG_X)

    DG_X = np.array(df["DG"]).reshape(-1, 1)
    df["DG"] = scaler.fit_transform(DG_X)

    SG_DG_SUM_X = np.array(df["SG_DG_SUM"]).reshape(-1, 1)
    df["SG_DG_SUM"] = scaler.fit_transform(SG_DG_SUM_X)

    SG_DG_PRO_X = np.array(df["SG_DG_PRO"]).reshape(-1, 1)
    df["SG_DG_PRO"] = scaler.fit_transform(SG_DG_PRO_X)

    SG_DG_SUB_X = np.array(df["SG_DG_SUB"]).reshape(-1, 1)
    df["SG_DG_SUB"] = scaler.fit_transform(SG_DG_SUB_X)

    EV_X = np.array(df["EV"]).reshape(-1, 1)
    df["EV"] = scaler.fit_transform(EV_X)

    SG_EV_SUM_X = np.array(df["SG_EV_SUM"]).reshape(-1, 1)
    df["SG_EV_SUM"] = scaler.fit_transform(SG_EV_SUM_X)

    SG_EV_PRO_X = np.array(df["SG_EV_PRO"]).reshape(-1, 1)
    df["SG_EV_PRO"] = scaler.fit_transform(SG_EV_PRO_X)

    SG_EV_SUB_X = np.array(df["SG_EV_SUB"]).reshape(-1, 1)
    df["SG_EV_SUB"] = scaler.fit_transform(SG_EV_SUB_X)

    DG_EV_SUM_X = np.array(df["DG_EV_SUM"]).reshape(-1, 1)
    df["DG_EV_SUM"] = scaler.fit_transform(DG_EV_SUM_X)

    DG_EV_PRO_X = np.array(df["DG_EV_PRO"]).reshape(-1, 1)
    df["DG_EV_PRO"] = scaler.fit_transform(DG_EV_PRO_X)

    DG_EV_SUB_X = np.array(df["DG_EV_SUB"]).reshape(-1, 1)
    df["DG_EV_SUB"] = scaler.fit_transform(DG_EV_SUB_X)

    INFO_X = np.array(df["INFO"]).reshape(-1, 1)
    df["INFO"] = scaler.fit_transform(INFO_X)

    SG_INFO_SUM_X = np.array(df["SG_INFO_SUM"]).reshape(-1, 1)
    df["SG_INFO_SUM"] = scaler.fit_transform(SG_INFO_SUM_X)

    SG_INFO_PRO_X = np.array(df["SG_INFO_PRO"]).reshape(-1, 1)
    df["SG_INFO_PRO"] = scaler.fit_transform(SG_INFO_PRO_X)

    SG_INFO_SUB_X = np.array(df["SG_INFO_SUB"]).reshape(-1, 1)
    df["SG_INFO_SUB"] = scaler.fit_transform(SG_INFO_SUB_X)

    DG_INFO_SUM_X = np.array(df["DG_INFO_SUM"]).reshape(-1, 1)
    df["DG_INFO_SUM"] = scaler.fit_transform(DG_INFO_SUM_X)

    DG_INFO_PRO_X = np.array(df["DG_INFO_PRO"]).reshape(-1, 1)
    df["DG_INFO_PRO"] = scaler.fit_transform(DG_INFO_PRO_X)

    DG_INFO_SUB_X = np.array(df["DG_INFO_SUB"]).reshape(-1, 1)
    df["DG_INFO_SUB"] = scaler.fit_transform(DG_INFO_SUB_X)

    EV_INFO_SUM_X = np.array(df["EV_INFO_SUM"]).reshape(-1, 1)
    df["EV_INFO_SUM"] = scaler.fit_transform(EV_INFO_SUM_X)

    EV_INFO_PRO_X = np.array(df["EV_INFO_PRO"]).reshape(-1, 1)
    df["EV_INFO_PRO"] = scaler.fit_transform(EV_INFO_PRO_X)

    EV_INFO_SUB_X = np.array(df["EV_INFO_SUB"]).reshape(-1, 1)
    df["EV_INFO_SUB"] = scaler.fit_transform(EV_INFO_SUB_X)

    LAC_X = np.array(df["LAC"]).reshape(-1, 1)
    df["LAC"] = scaler.fit_transform(LAC_X)

    SG_LAC_SUM_X = np.array(df["SG_LAC_SUM"]).reshape(-1, 1)
    df["SG_LAC_SUM"] = scaler.fit_transform(SG_LAC_SUM_X)

    SG_LAC_PRO_X = np.array(df["SG_LAC_PRO"]).reshape(-1, 1)
    df["SG_LAC_PRO"] = scaler.fit_transform(SG_LAC_PRO_X)

    SG_LAC_SUB_X = np.array(df["SG_LAC_SUB"]).reshape(-1, 1)
    df["SG_LAC_SUB"] = scaler.fit_transform(SG_LAC_SUB_X)

    DG_LAC_SUM_X = np.array(df["DG_LAC_SUM"]).reshape(-1, 1)
    df["DG_LAC_SUM"] = scaler.fit_transform(DG_LAC_SUM_X)

    DG_LAC_PRO_X = np.array(df["DG_LAC_PRO"]).reshape(-1, 1)
    df["DG_LAC_PRO"] = scaler.fit_transform(DG_LAC_PRO_X)

    DG_LAC_SUB_X = np.array(df["DG_LAC_SUB"]).reshape(-1, 1)
    df["DG_LAC_SUB"] = scaler.fit_transform(DG_LAC_SUB_X)

    EV_LAC_SUM_X = np.array(df["EV_LAC_SUM"]).reshape(-1, 1)
    df["EV_LAC_SUM"] = scaler.fit_transform(EV_LAC_SUM_X)

    EV_LAC_PRO_X = np.array(df["EV_LAC_PRO"]).reshape(-1, 1)
    df["EV_LAC_PRO"] = scaler.fit_transform(EV_LAC_PRO_X)

    EV_LAC_SUB_X = np.array(df["EV_LAC_SUB"]).reshape(-1, 1)
    df["EV_LAC_SUB"] = scaler.fit_transform(EV_LAC_SUB_X)

    INFO_LAC_SUM_X = np.array(df["INFO_LAC_SUM"]).reshape(-1, 1)
    df["INFO_LAC_SUM"] = scaler.fit_transform(INFO_LAC_SUM_X)

    INFO_LAC_PRO_X = np.array(df["INFO_LAC_PRO"]).reshape(-1, 1)
    df["INFO_LAC_PRO"] = scaler.fit_transform(INFO_LAC_PRO_X)

    INFO_LAC_SUB_X = np.array(df["INFO_LAC_SUB"]).reshape(-1, 1)
    df["INFO_LAC_SUB"] = scaler.fit_transform(INFO_LAC_SUB_X)

    BNESS_X = np.array(df["BNESS"]).reshape(-1, 1)
    df["BNESS"] = scaler.fit_transform(BNESS_X)

    SG_BNESS_SUM_X = np.array(df["SG_BNESS_SUM"]).reshape(-1, 1)
    df["SG_BNESS_SUM"] = scaler.fit_transform(SG_BNESS_SUM_X)

    SG_BNESS_PRO_X = np.array(df["SG_BNESS_PRO"]).reshape(-1, 1)
    df["SG_BNESS_PRO"] = scaler.fit_transform(SG_BNESS_PRO_X)

    SG_BNESS_SUB_X = np.array(df["SG_BNESS_SUB"]).reshape(-1, 1)
    df["SG_BNESS_SUB"] = scaler.fit_transform(SG_BNESS_SUB_X)

    DG_BNESS_SUM_X = np.array(df["DG_BNESS_SUM"]).reshape(-1, 1)
    df["DG_BNESS_SUM"] = scaler.fit_transform(DG_BNESS_SUM_X)

    DG_BNESS_PRO_X = np.array(df["DG_BNESS_PRO"]).reshape(-1, 1)
    df["DG_BNESS_PRO"] = scaler.fit_transform(DG_BNESS_PRO_X)

    DG_BNESS_SUB_X = np.array(df["DG_BNESS_SUB"]).reshape(-1, 1)
    df["DG_BNESS_SUB"] = scaler.fit_transform(DG_BNESS_SUB_X)

    EV_BNESS_SUM_X = np.array(df["EV_BNESS_SUM"]).reshape(-1, 1)
    df["EV_BNESS_SUM"] = scaler.fit_transform(EV_BNESS_SUM_X)

    EV_BNESS_PRO_X = np.array(df["EV_BNESS_PRO"]).reshape(-1, 1)
    df["EV_BNESS_PRO"] = scaler.fit_transform(EV_BNESS_PRO_X)

    EV_BNESS_SUB_X = np.array(df["EV_BNESS_SUB"]).reshape(-1, 1)
    df["EV_BNESS_SUB"] = scaler.fit_transform(EV_BNESS_SUB_X)

    INFO_BNESS_SUM_X = np.array(df["INFO_BNESS_SUM"]).reshape(-1, 1)
    df["INFO_BNESS_SUM"] = scaler.fit_transform(INFO_BNESS_SUM_X)

    INFO_BNESS_PRO_X = np.array(df["INFO_BNESS_PRO"]).reshape(-1, 1)
    df["INFO_BNESS_PRO"] = scaler.fit_transform(INFO_BNESS_PRO_X)

    INFO_BNESS_SUB_X = np.array(df["INFO_BNESS_SUB"]).reshape(-1, 1)
    df["INFO_BNESS_SUB"] = scaler.fit_transform(INFO_BNESS_SUB_X)

    LAC_BNESS_SUM_X = np.array(df["LAC_BNESS_SUM"]).reshape(-1, 1)
    df["LAC_BNESS_SUM"] = scaler.fit_transform(LAC_BNESS_SUM_X)

    LAC_BNESS_PRO_X = np.array(df["LAC_BNESS_PRO"]).reshape(-1, 1)
    df["LAC_BNESS_PRO"] = scaler.fit_transform(LAC_BNESS_PRO_X)

    LAC_BNESS_SUB_X = np.array(df["LAC_BNESS_SUB"]).reshape(-1, 1)
    df["LAC_BNESS_SUB"] = scaler.fit_transform(LAC_BNESS_SUB_X)

    CNESS_X = np.array(df["CNESS"]).reshape(-1, 1)
    df["CNESS"] = scaler.fit_transform(CNESS_X)

    SG_CNESS_SUM_X = np.array(df["SG_CNESS_SUM"]).reshape(-1, 1)
    df["SG_CNESS_SUM"] = scaler.fit_transform(SG_CNESS_SUM_X)

    SG_CNESS_PRO_X = np.array(df["SG_CNESS_PRO"]).reshape(-1, 1)
    df["SG_CNESS_PRO"] = scaler.fit_transform(SG_CNESS_PRO_X)

    SG_CNESS_SUB_X = np.array(df["SG_CNESS_SUB"]).reshape(-1, 1)
    df["SG_CNESS_SUB"] = scaler.fit_transform(SG_CNESS_SUB_X)

    DG_CNESS_SUM_X = np.array(df["DG_CNESS_SUM"]).reshape(-1, 1)
    df["DG_CNESS_SUM"] = scaler.fit_transform(DG_CNESS_SUM_X)

    DG_CNESS_PRO_X = np.array(df["DG_CNESS_PRO"]).reshape(-1, 1)
    df["DG_CNESS_PRO"] = scaler.fit_transform(DG_CNESS_PRO_X)

    DG_CNESS_SUB_X = np.array(df["DG_CNESS_SUB"]).reshape(-1, 1)
    df["DG_CNESS_SUB"] = scaler.fit_transform(DG_CNESS_SUB_X)

    EV_CNESS_SUM_X = np.array(df["EV_CNESS_SUM"]).reshape(-1, 1)
    df["EV_CNESS_SUM"] = scaler.fit_transform(EV_CNESS_SUM_X)

    EV_CNESS_PRO_X = np.array(df["EV_CNESS_PRO"]).reshape(-1, 1)
    df["EV_CNESS_PRO"] = scaler.fit_transform(EV_CNESS_PRO_X)

    EV_CNESS_SUB_X = np.array(df["EV_CNESS_SUB"]).reshape(-1, 1)
    df["EV_CNESS_SUB"] = scaler.fit_transform(EV_CNESS_SUB_X)

    INFO_CNESS_SUM_X = np.array(df["INFO_CNESS_SUM"]).reshape(-1, 1)
    df["INFO_CNESS_SUM"] = scaler.fit_transform(INFO_CNESS_SUM_X)

    INFO_CNESS_PRO_X = np.array(df["INFO_CNESS_PRO"]).reshape(-1, 1)
    df["INFO_CNESS_PRO"] = scaler.fit_transform(INFO_CNESS_PRO_X)

    INFO_CNESS_SUB_X = np.array(df["INFO_CNESS_SUB"]).reshape(-1, 1)
    df["INFO_CNESS_SUB"] = scaler.fit_transform(INFO_CNESS_SUB_X)

    LAC_CNESS_SUM_X = np.array(df["LAC_CNESS_SUM"]).reshape(-1, 1)
    df["LAC_CNESS_SUM"] = scaler.fit_transform(LAC_CNESS_SUM_X)

    LAC_CNESS_PRO_X = np.array(df["LAC_CNESS_PRO"]).reshape(-1, 1)
    df["LAC_CNESS_PRO"] = scaler.fit_transform(LAC_CNESS_PRO_X)

    LAC_CNESS_SUB_X = np.array(df["LAC_CNESS_SUB"]).reshape(-1, 1)
    df["LAC_CNESS_SUB"] = scaler.fit_transform(LAC_CNESS_SUB_X)

    BNESS_CNESS_SUM_X = np.array(df["BNESS_CNESS_SUM"]).reshape(-1, 1)
    df["BNESS_CNESS_SUM"] = scaler.fit_transform(BNESS_CNESS_SUM_X)

    BNESS_CNESS_PRO_X = np.array(df["BNESS_CNESS_PRO"]).reshape(-1, 1)
    df["BNESS_CNESS_PRO"] = scaler.fit_transform(BNESS_CNESS_PRO_X)

    BNESS_CNESS_SUB_X = np.array(df["BNESS_CNESS_SUB"]).reshape(-1, 1)
    df["BNESS_CNESS_SUB"] = scaler.fit_transform(BNESS_CNESS_SUB_X)

    NET_X = np.array(df["NET"]).reshape(-1, 1)
    df["NET"] = scaler.fit_transform(NET_X)

    SG_NET_SUM_X = np.array(df["SG_NET_SUM"]).reshape(-1, 1)
    df["SG_NET_SUM"] = scaler.fit_transform(SG_NET_SUM_X)

    SG_NET_PRO_X = np.array(df["SG_NET_PRO"]).reshape(-1, 1)
    df["SG_NET_PRO"] = scaler.fit_transform(SG_NET_PRO_X)

    SG_NET_SUB_X = np.array(df["SG_NET_SUB"]).reshape(-1, 1)
    df["SG_NET_SUB"] = scaler.fit_transform(SG_NET_SUB_X)

    DG_NET_SUM_X = np.array(df["DG_NET_SUM"]).reshape(-1, 1)
    df["DG_NET_SUM"] = scaler.fit_transform(DG_NET_SUM_X)

    DG_NET_PRO_X = np.array(df["DG_NET_PRO"]).reshape(-1, 1)
    df["DG_NET_PRO"] = scaler.fit_transform(DG_NET_PRO_X)

    DG_NET_SUB_X = np.array(df["DG_NET_SUB"]).reshape(-1, 1)
    df["DG_NET_SUB"] = scaler.fit_transform(DG_NET_SUB_X)

    EV_NET_SUM_X = np.array(df["EV_NET_SUM"]).reshape(-1, 1)
    df["EV_NET_SUM"] = scaler.fit_transform(EV_NET_SUM_X)

    EV_NET_PRO_X = np.array(df["EV_NET_PRO"]).reshape(-1, 1)
    df["EV_NET_PRO"] = scaler.fit_transform(EV_NET_PRO_X)

    EV_NET_SUB_X = np.array(df["EV_NET_SUB"]).reshape(-1, 1)
    df["EV_NET_SUB"] = scaler.fit_transform(EV_NET_SUB_X)

    INFO_NET_SUM_X = np.array(df["INFO_NET_SUM"]).reshape(-1, 1)
    df["INFO_NET_SUM"] = scaler.fit_transform(INFO_NET_SUM_X)

    INFO_NET_PRO_X = np.array(df["INFO_NET_PRO"]).reshape(-1, 1)
    df["INFO_NET_PRO"] = scaler.fit_transform(INFO_NET_PRO_X)

    INFO_NET_SUB_X = np.array(df["INFO_NET_SUB"]).reshape(-1, 1)
    df["INFO_NET_SUB"] = scaler.fit_transform(INFO_NET_SUB_X)

    LAC_NET_SUM_X = np.array(df["LAC_NET_SUM"]).reshape(-1, 1)
    df["LAC_NET_SUM"] = scaler.fit_transform(LAC_NET_SUM_X)

    LAC_NET_PRO_X = np.array(df["LAC_NET_PRO"]).reshape(-1, 1)
    df["LAC_NET_PRO"] = scaler.fit_transform(LAC_NET_PRO_X)

    LAC_NET_SUB_X = np.array(df["LAC_NET_SUB"]).reshape(-1, 1)
    df["LAC_NET_SUB"] = scaler.fit_transform(LAC_NET_SUB_X)

    BNESS_NET_SUM_X = np.array(df["BNESS_NET_SUM"]).reshape(-1, 1)
    df["BNESS_NET_SUM"] = scaler.fit_transform(BNESS_NET_SUM_X)

    BNESS_NET_PRO_X = np.array(df["BNESS_NET_PRO"]).reshape(-1, 1)
    df["BNESS_NET_PRO"] = scaler.fit_transform(BNESS_NET_PRO_X)

    BNESS_NET_SUB_X = np.array(df["BNESS_NET_SUB"]).reshape(-1, 1)
    df["BNESS_NET_SUB"] = scaler.fit_transform(BNESS_NET_SUB_X)

    CNESS_NET_SUM_X = np.array(df["CNESS_NET_SUM"]).reshape(-1, 1)
    df["CNESS_NET_SUM"] = scaler.fit_transform(CNESS_NET_SUM_X)

    CNESS_NET_PRO_X = np.array(df["CNESS_NET_PRO"]).reshape(-1, 1)
    df["CNESS_NET_PRO"] = scaler.fit_transform(CNESS_NET_PRO_X)

    CNESS_NET_SUB_X = np.array(df["CNESS_NET_SUB"]).reshape(-1, 1)
    df["CNESS_NET_SUB"] = scaler.fit_transform(CNESS_NET_SUB_X)

    F_X = np.array(df["GE_T1"]).reshape(-1, 1)
    df["GE_T1"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T1_GE_T2_SUM"]).reshape(-1, 1)
    df["GE_T1_GE_T2_SUM"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T1_GE_T2_PRO"]).reshape(-1, 1)
    df["GE_T1_GE_T2_PRO"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T1_GE_T2_SUB"]).reshape(-1, 1)
    df["GE_T1_GE_T2_SUB"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T1_GE_T3_SUM"]).reshape(-1, 1)
    df["GE_T1_GE_T3_SUM"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T1_GE_T3_PRO"]).reshape(-1, 1)
    df["GE_T1_GE_T3_PRO"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T1_GE_T3_SUB"]).reshape(-1, 1)
    df["GE_T1_GE_T3_SUB"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T1_GE_T4_SUM"]).reshape(-1, 1)
    df["GE_T1_GE_T4_SUM"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T1_GE_T4_PRO"]).reshape(-1, 1)
    df["GE_T1_GE_T4_PRO"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T1_GE_T4_SUB"]).reshape(-1, 1)
    df["GE_T1_GE_T4_SUB"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T1_GE_T5_SUM"]).reshape(-1, 1)
    df["GE_T1_GE_T5_SUM"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T1_GE_T5_PRO"]).reshape(-1, 1)
    df["GE_T1_GE_T5_PRO"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T1_GE_T5_SUB"]).reshape(-1, 1)
    df["GE_T1_GE_T5_SUB"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T1_GE_T6_SUM"]).reshape(-1, 1)
    df["GE_T1_GE_T6_SUM"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T1_GE_T6_PRO"]).reshape(-1, 1)
    df["GE_T1_GE_T6_PRO"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T1_GE_T6_SUB"]).reshape(-1, 1)
    df["GE_T1_GE_T6_SUB"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T1_GE_T7_SUM"]).reshape(-1, 1)
    df["GE_T1_GE_T7_SUM"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T1_GE_T7_PRO"]).reshape(-1, 1)
    df["GE_T1_GE_T7_PRO"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T1_GE_T7_SUB"]).reshape(-1, 1)
    df["GE_T1_GE_T7_SUB"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T1_GE_T8_SUM"]).reshape(-1, 1)
    df["GE_T1_GE_T8_SUM"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T1_GE_T8_PRO"]).reshape(-1, 1)
    df["GE_T1_GE_T8_PRO"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T1_GE_T8_SUB"]).reshape(-1, 1)
    df["GE_T1_GE_T8_SUB"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T2"]).reshape(-1, 1)
    df["GE_T2"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T2_GE_T3_SUM"]).reshape(-1, 1)
    df["GE_T2_GE_T3_SUM"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T2_GE_T3_PRO"]).reshape(-1, 1)
    df["GE_T2_GE_T3_PRO"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T2_GE_T3_SUB"]).reshape(-1, 1)
    df["GE_T2_GE_T3_SUB"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T2_GE_T4_SUM"]).reshape(-1, 1)
    df["GE_T2_GE_T4_SUM"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T2_GE_T4_PRO"]).reshape(-1, 1)
    df["GE_T2_GE_T4_PRO"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T2_GE_T4_SUB"]).reshape(-1, 1)
    df["GE_T2_GE_T4_SUB"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T2_GE_T5_SUM"]).reshape(-1, 1)
    df["GE_T2_GE_T5_SUM"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T2_GE_T5_PRO"]).reshape(-1, 1)
    df["GE_T2_GE_T5_PRO"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T2_GE_T5_SUB"]).reshape(-1, 1)
    df["GE_T2_GE_T5_SUB"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T2_GE_T6_SUM"]).reshape(-1, 1)
    df["GE_T2_GE_T6_SUM"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T2_GE_T6_PRO"]).reshape(-1, 1)
    df["GE_T2_GE_T6_PRO"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T2_GE_T6_SUB"]).reshape(-1, 1)
    df["GE_T2_GE_T6_SUB"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T2_GE_T7_SUM"]).reshape(-1, 1)
    df["GE_T2_GE_T7_SUM"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T2_GE_T7_PRO"]).reshape(-1, 1)
    df["GE_T2_GE_T7_PRO"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T2_GE_T7_SUB"]).reshape(-1, 1)
    df["GE_T2_GE_T7_SUB"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T2_GE_T8_SUM"]).reshape(-1, 1)
    df["GE_T2_GE_T8_SUM"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T2_GE_T8_PRO"]).reshape(-1, 1)
    df["GE_T2_GE_T8_PRO"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T2_GE_T8_SUB"]).reshape(-1, 1)
    df["GE_T2_GE_T8_SUB"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T3"]).reshape(-1, 1)
    df["GE_T3"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T3_GE_T4_SUM"]).reshape(-1, 1)
    df["GE_T3_GE_T4_SUM"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T3_GE_T4_PRO"]).reshape(-1, 1)
    df["GE_T3_GE_T4_PRO"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T3_GE_T4_SUB"]).reshape(-1, 1)
    df["GE_T3_GE_T4_SUB"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T3_GE_T5_SUM"]).reshape(-1, 1)
    df["GE_T3_GE_T5_SUM"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T3_GE_T5_PRO"]).reshape(-1, 1)
    df["GE_T3_GE_T5_PRO"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T3_GE_T5_SUB"]).reshape(-1, 1)
    df["GE_T3_GE_T5_SUB"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T3_GE_T6_SUM"]).reshape(-1, 1)
    df["GE_T3_GE_T6_SUM"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T3_GE_T6_PRO"]).reshape(-1, 1)
    df["GE_T3_GE_T6_PRO"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T3_GE_T6_SUB"]).reshape(-1, 1)
    df["GE_T3_GE_T6_SUB"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T3_GE_T7_SUM"]).reshape(-1, 1)
    df["GE_T3_GE_T7_SUM"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T3_GE_T7_PRO"]).reshape(-1, 1)
    df["GE_T3_GE_T7_PRO"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T3_GE_T7_SUB"]).reshape(-1, 1)
    df["GE_T3_GE_T7_SUB"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T3_GE_T8_SUM"]).reshape(-1, 1)
    df["GE_T3_GE_T8_SUM"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T3_GE_T8_PRO"]).reshape(-1, 1)
    df["GE_T3_GE_T8_PRO"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T3_GE_T8_SUB"]).reshape(-1, 1)
    df["GE_T3_GE_T8_SUB"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T4"]).reshape(-1, 1)
    df["GE_T4"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T4_GE_T5_SUM"]).reshape(-1, 1)
    df["GE_T4_GE_T5_SUM"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T4_GE_T5_PRO"]).reshape(-1, 1)
    df["GE_T4_GE_T5_PRO"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T4_GE_T5_SUB"]).reshape(-1, 1)
    df["GE_T4_GE_T5_SUB"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T4_GE_T6_SUM"]).reshape(-1, 1)
    df["GE_T4_GE_T6_SUM"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T4_GE_T6_PRO"]).reshape(-1, 1)
    df["GE_T4_GE_T6_PRO"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T4_GE_T6_SUB"]).reshape(-1, 1)
    df["GE_T4_GE_T6_SUB"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T4_GE_T7_SUM"]).reshape(-1, 1)
    df["GE_T4_GE_T7_SUM"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T4_GE_T7_PRO"]).reshape(-1, 1)
    df["GE_T4_GE_T7_PRO"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T4_GE_T7_SUB"]).reshape(-1, 1)
    df["GE_T4_GE_T7_SUB"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T4_GE_T8_SUM"]).reshape(-1, 1)
    df["GE_T4_GE_T8_SUM"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T4_GE_T8_PRO"]).reshape(-1, 1)
    df["GE_T4_GE_T8_PRO"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T4_GE_T8_SUB"]).reshape(-1, 1)
    df["GE_T4_GE_T8_SUB"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T5"]).reshape(-1, 1)
    df["GE_T5"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T5_GE_T6_SUM"]).reshape(-1, 1)
    df["GE_T5_GE_T6_SUM"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T5_GE_T6_PRO"]).reshape(-1, 1)
    df["GE_T5_GE_T6_PRO"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T5_GE_T6_SUB"]).reshape(-1, 1)
    df["GE_T5_GE_T6_SUB"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T5_GE_T7_SUM"]).reshape(-1, 1)
    df["GE_T5_GE_T7_SUM"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T5_GE_T7_PRO"]).reshape(-1, 1)
    df["GE_T5_GE_T7_PRO"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T5_GE_T7_SUB"]).reshape(-1, 1)
    df["GE_T5_GE_T7_SUB"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T5_GE_T8_SUM"]).reshape(-1, 1)
    df["GE_T5_GE_T8_SUM"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T5_GE_T8_PRO"]).reshape(-1, 1)
    df["GE_T5_GE_T8_PRO"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T5_GE_T8_SUB"]).reshape(-1, 1)
    df["GE_T5_GE_T8_SUB"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T6"]).reshape(-1, 1)
    df["GE_T6"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T6_GE_T7_SUM"]).reshape(-1, 1)
    df["GE_T6_GE_T7_SUM"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T6_GE_T7_PRO"]).reshape(-1, 1)
    df["GE_T6_GE_T7_PRO"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T6_GE_T7_SUB"]).reshape(-1, 1)
    df["GE_T6_GE_T7_SUB"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T6_GE_T8_SUM"]).reshape(-1, 1)
    df["GE_T6_GE_T8_SUM"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T6_GE_T8_PRO"]).reshape(-1, 1)
    df["GE_T6_GE_T8_PRO"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T6_GE_T8_SUB"]).reshape(-1, 1)
    df["GE_T6_GE_T8_SUB"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T7"]).reshape(-1, 1)
    df["GE_T7"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T7_GE_T8_SUM"]).reshape(-1, 1)
    df["GE_T7_GE_T8_SUM"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T7_GE_T8_PRO"]).reshape(-1, 1)
    df["GE_T7_GE_T8_PRO"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T7_GE_T8_SUB"]).reshape(-1, 1)
    df["GE_T7_GE_T8_SUB"] = scaler.fit_transform(F_X)

    F_X = np.array(df["GE_T8"]).reshape(-1, 1)
    df["GE_T8"] = scaler.fit_transform(F_X)

    c = 0
    # Calcualte PE and KE of all initial molecules using SVM
    for row in range(TotalMolecule):

        features.clear()

        if copy_molecules[row][0] == 1:
            features.append('SG')
        if copy_molecules[row][1] == 1:
            features.append('DG')
        if copy_molecules[row][2] == 1:
            features.append('SG_DG_SUM')
        if copy_molecules[row][3] == 1:
            features.append('SG_DG_PRO')
        if copy_molecules[row][4] == 1:
            features.append('SG_DG_SUB')
        if copy_molecules[row][5] == 1:
            features.append('EV')
        if copy_molecules[row][6] == 1:
            features.append('SG_EV_SUM')
        if copy_molecules[row][7] == 1:
            features.append('SG_EV_PRO')
        if copy_molecules[row][8] == 1:
            features.append('SG_EV_SUB')
        if copy_molecules[row][9] == 1:
            features.append('DG_EV_SUM')
        if copy_molecules[row][10] == 1:
            features.append('DG_EV_PRO')
        if copy_molecules[row][11] == 1:
            features.append('DG_EV_SUB')
        if copy_molecules[row][12] == 1:
            features.append('INFO')
        if copy_molecules[row][13] == 1:
            features.append('SG_INFO_SUM')
        if copy_molecules[row][14] == 1:
            features.append('SG_INFO_PRO')
        if copy_molecules[row][15] == 1:
            features.append('SG_INFO_SUB')
        if copy_molecules[row][16] == 1:
            features.append('DG_INFO_SUM')
        if copy_molecules[row][17] == 1:
            features.append('DG_INFO_PRO')
        if copy_molecules[row][18] == 1:
            features.append('DG_INFO_SUB')
        if copy_molecules[row][19] == 1:
            features.append('EV_INFO_SUM')
        if copy_molecules[row][20] == 1:
            features.append('EV_INFO_PRO')
        if copy_molecules[row][21] == 1:
            features.append('EV_INFO_SUB')
        if copy_molecules[row][22] == 1:
            features.append('LAC')
        if copy_molecules[row][23] == 1:
            features.append('SG_LAC_SUM')
        if copy_molecules[row][24] == 1:
            features.append('SG_LAC_PRO')
        if copy_molecules[row][25] == 1:
            features.append('SG_LAC_SUB')
        if copy_molecules[row][26] == 1:
            features.append('DG_LAC_SUM')
        if copy_molecules[row][27] == 1:
            features.append('DG_LAC_PRO')
        if copy_molecules[row][28] == 1:
            features.append('DG_LAC_SUB')
        if copy_molecules[row][29] == 1:
            features.append('EV_LAC_SUM')
        if copy_molecules[row][30] == 1:
            features.append('EV_LAC_PRO')
        if copy_molecules[row][31] == 1:
            features.append('EV_LAC_SUB')
        if copy_molecules[row][32] == 1:
            features.append('INFO_LAC_SUM')
        if copy_molecules[row][33] == 1:
            features.append('INFO_LAC_PRO')
        if copy_molecules[row][34] == 1:
            features.append('INFO_LAC_SUB')
        if copy_molecules[row][35] == 1:
            features.append('BNESS')
        if copy_molecules[row][36] == 1:
            features.append('SG_BNESS_SUM')
        if copy_molecules[row][37] == 1:
            features.append('SG_BNESS_PRO')
        if copy_molecules[row][38] == 1:
            features.append('SG_BNESS_SUB')
        if copy_molecules[row][39] == 1:
            features.append('DG_BNESS_SUM')
        if copy_molecules[row][40] == 1:
            features.append('DG_BNESS_PRO')
        if copy_molecules[row][41] == 1:
            features.append('DG_BNESS_SUB')
        if copy_molecules[row][42] == 1:
            features.append('EV_BNESS_SUM')
        if copy_molecules[row][43] == 1:
            features.append('EV_BNESS_PRO')
        if copy_molecules[row][44] == 1:
            features.append('EV_BNESS_SUB')
        if copy_molecules[row][45] == 1:
            features.append('INFO_BNESS_SUM')
        if copy_molecules[row][46] == 1:
            features.append('INFO_BNESS_PRO')
        if copy_molecules[row][47] == 1:
            features.append('INFO_BNESS_SUB')
        if copy_molecules[row][48] == 1:
            features.append('LAC_BNESS_SUM')
        if copy_molecules[row][49] == 1:
            features.append('LAC_BNESS_PRO')
        if copy_molecules[row][50] == 1:
            features.append('LAC_BNESS_SUB')
        if copy_molecules[row][51] == 1:
            features.append('CNESS')
        if copy_molecules[row][52] == 1:
            features.append('SG_CNESS_SUM')
        if copy_molecules[row][53] == 1:
            features.append('SG_CNESS_PRO')
        if copy_molecules[row][54] == 1:
            features.append('SG_CNESS_SUB')
        if copy_molecules[row][55] == 1:
            features.append('DG_CNESS_SUM')
        if copy_molecules[row][56] == 1:
            features.append('DG_CNESS_PRO')
        if copy_molecules[row][57] == 1:
            features.append('DG_CNESS_SUB')
        if copy_molecules[row][58] == 1:
            features.append('EV_CNESS_SUM')
        if copy_molecules[row][59] == 1:
            features.append('EV_CNESS_PRO')
        if copy_molecules[row][60] == 1:
            features.append('EV_CNESS_SUB')
        if copy_molecules[row][61] == 1:
            features.append('INFO_CNESS_SUM')
        if copy_molecules[row][62] == 1:
            features.append('INFO_CNESS_PRO')
        if copy_molecules[row][63] == 1:
            features.append('INFO_CNESS_SUB')
        if copy_molecules[row][64] == 1:
            features.append('LAC_CNESS_SUM')
        if copy_molecules[row][65] == 1:
            features.append('LAC_CNESS_PRO')
        if copy_molecules[row][66] == 1:
            features.append('LAC_CNESS_SUB')
        if copy_molecules[row][67] == 1:
            features.append('BNESS_CNESS_SUM')
        if copy_molecules[row][68] == 1:
            features.append('BNESS_CNESS_PRO')
        if copy_molecules[row][69] == 1:
            features.append('BNESS_CNESS_SUB')
        if copy_molecules[row][70] == 1:
            features.append('NET')
        if copy_molecules[row][71] == 1:
            features.append('SG_NET_SUM')
        if copy_molecules[row][72] == 1:
            features.append('SG_NET_PRO')
        if copy_molecules[row][73] == 1:
            features.append('SG_NET_SUB')
        if copy_molecules[row][74] == 1:
            features.append('DG_NET_SUM')
        if copy_molecules[row][75] == 1:
            features.append('DG_NET_PRO')
        if copy_molecules[row][76] == 1:
            features.append('DG_NET_SUB')
        if copy_molecules[row][77] == 1:
            features.append('EV_NET_SUM')
        if copy_molecules[row][78] == 1:
            features.append('EV_NET_PRO')
        if copy_molecules[row][79] == 1:
            features.append('EV_NET_SUB')
        if copy_molecules[row][80] == 1:
            features.append('INFO_NET_SUM')
        if copy_molecules[row][81] == 1:
            features.append('INFO_NET_PRO')
        if copy_molecules[row][82] == 1:
            features.append('INFO_NET_SUB')
        if copy_molecules[row][83] == 1:
            features.append('LAC_NET_SUM')
        if copy_molecules[row][84] == 1:
            features.append('LAC_NET_PRO')
        if copy_molecules[row][85] == 1:
            features.append('LAC_NET_SUB')
        if copy_molecules[row][86] == 1:
            features.append('BNESS_NET_SUM')
        if copy_molecules[row][87] == 1:
            features.append('BNESS_NET_PRO')
        if copy_molecules[row][88] == 1:
            features.append('BNESS_NET_SUB')
        if copy_molecules[row][89] == 1:
            features.append('CNESS_NET_SUM')
        if copy_molecules[row][90] == 1:
            features.append('CNESS_NET_PRO')
        if copy_molecules[row][91] == 1:
            features.append('CNESS_NET_SUB')
        if copy_molecules[row][92] == 1:
            features.append('GE_T1')
        if copy_molecules[row][93] == 1:
            features.append('GE_T1_GE_T2_SUM')
        if copy_molecules[row][94] == 1:
            features.append('GE_T1_GE_T2_PRO')
        if copy_molecules[row][95] == 1:
            features.append('GE_T1_GE_T2_SUB')
        if copy_molecules[row][96] == 1:
            features.append('GE_T1_GE_T3_SUM')
        if copy_molecules[row][97] == 1:
            features.append('GE_T1_GE_T3_PRO')
        if copy_molecules[row][98] == 1:
            features.append('GE_T1_GE_T3_SUB')
        if copy_molecules[row][99] == 1:
            features.append('GE_T1_GE_T4_SUM')
        if copy_molecules[row][100] == 1:
            features.append('GE_T1_GE_T4_PRO')
        if copy_molecules[row][101] == 1:
            features.append('GE_T1_GE_T4_SUB')
        if copy_molecules[row][102] == 1:
            features.append('GE_T1_GE_T5_SUM')
        if copy_molecules[row][103] == 1:
            features.append('GE_T1_GE_T5_PRO')
        if copy_molecules[row][104] == 1:
            features.append('GE_T1_GE_T5_SUB')
        if copy_molecules[row][105] == 1:
            features.append('GE_T1_GE_T6_SUM')
        if copy_molecules[row][106] == 1:
            features.append('GE_T1_GE_T6_PRO')
        if copy_molecules[row][107] == 1:
            features.append('GE_T1_GE_T6_SUB')
        if copy_molecules[row][108] == 1:
            features.append('GE_T1_GE_T7_SUM')
        if copy_molecules[row][109] == 1:
            features.append('GE_T1_GE_T7_PRO')
        if copy_molecules[row][110] == 1:
            features.append('GE_T1_GE_T7_SUB')
        if copy_molecules[row][111] == 1:
            features.append('GE_T1_GE_T8_SUM')
        if copy_molecules[row][112] == 1:
            features.append('GE_T1_GE_T8_PRO')
        if copy_molecules[row][113] == 1:
            features.append('GE_T1_GE_T8_SUB')
        if copy_molecules[row][114] == 1:
            features.append('GE_T2')
        if copy_molecules[row][115] == 1:
            features.append('GE_T2_GE_T3_SUM')
        if copy_molecules[row][116] == 1:
            features.append('GE_T2_GE_T3_PRO')
        if copy_molecules[row][117] == 1:
            features.append('GE_T2_GE_T3_SUB')
        if copy_molecules[row][118] == 1:
            features.append('GE_T2_GE_T4_SUM')
        if copy_molecules[row][119] == 1:
            features.append('GE_T2_GE_T4_PRO')
        if copy_molecules[row][120] == 1:
            features.append('GE_T2_GE_T4_SUB')
        if copy_molecules[row][121] == 1:
            features.append('GE_T2_GE_T5_SUM')
        if copy_molecules[row][122] == 1:
            features.append('GE_T2_GE_T5_PRO')
        if copy_molecules[row][123] == 1:
            features.append('GE_T2_GE_T5_SUB')
        if copy_molecules[row][124] == 1:
            features.append('GE_T2_GE_T6_SUM')
        if copy_molecules[row][125] == 1:
            features.append('GE_T2_GE_T6_PRO')
        if copy_molecules[row][126] == 1:
            features.append('GE_T2_GE_T6_SUB')
        if copy_molecules[row][127] == 1:
            features.append('GE_T2_GE_T7_SUM')
        if copy_molecules[row][128] == 1:
            features.append('GE_T2_GE_T7_PRO')
        if copy_molecules[row][129] == 1:
            features.append('GE_T2_GE_T7_SUB')
        if copy_molecules[row][130] == 1:
            features.append('GE_T2_GE_T8_SUM')
        if copy_molecules[row][131] == 1:
            features.append('GE_T2_GE_T8_PRO')
        if copy_molecules[row][132] == 1:
            features.append('GE_T2_GE_T8_SUB')
        if copy_molecules[row][133] == 1:
            features.append('GE_T3')
        if copy_molecules[row][134] == 1:
            features.append('GE_T3_GE_T4_SUM')
        if copy_molecules[row][135] == 1:
            features.append('GE_T3_GE_T4_PRO')
        if copy_molecules[row][136] == 1:
            features.append('GE_T3_GE_T4_SUB')
        if copy_molecules[row][137] == 1:
            features.append('GE_T3_GE_T5_SUM')
        if copy_molecules[row][138] == 1:
            features.append('GE_T3_GE_T5_PRO')
        if copy_molecules[row][139] == 1:
            features.append('GE_T3_GE_T5_SUB')
        if copy_molecules[row][140] == 1:
            features.append('GE_T3_GE_T6_SUM')
        if copy_molecules[row][141] == 1:
            features.append('GE_T3_GE_T6_PRO')
        if copy_molecules[row][142] == 1:
            features.append('GE_T3_GE_T6_SUB')
        if copy_molecules[row][143] == 1:
            features.append('GE_T3_GE_T7_SUM')
        if copy_molecules[row][144] == 1:
            features.append('GE_T3_GE_T7_PRO')
        if copy_molecules[row][145] == 1:
            features.append('GE_T3_GE_T7_SUB')
        if copy_molecules[row][146] == 1:
            features.append('GE_T3_GE_T8_SUM')
        if copy_molecules[row][147] == 1:
            features.append('GE_T3_GE_T8_PRO')
        if copy_molecules[row][148] == 1:
            features.append('GE_T3_GE_T8_SUB')
        if copy_molecules[row][149] == 1:
            features.append('GE_T4')
        if copy_molecules[row][150] == 1:
            features.append('GE_T4_GE_T5_SUM')
        if copy_molecules[row][151] == 1:
            features.append('GE_T4_GE_T5_PRO')
        if copy_molecules[row][152] == 1:
            features.append('GE_T4_GE_T5_SUB')
        if copy_molecules[row][153] == 1:
            features.append('GE_T4_GE_T6_SUM')
        if copy_molecules[row][154] == 1:
            features.append('GE_T4_GE_T6_PRO')
        if copy_molecules[row][155] == 1:
            features.append('GE_T4_GE_T6_SUB')
        if copy_molecules[row][156] == 1:
            features.append('GE_T4_GE_T7_SUM')
        if copy_molecules[row][157] == 1:
            features.append('GE_T4_GE_T7_PRO')
        if copy_molecules[row][158] == 1:
            features.append('GE_T4_GE_T7_SUB')
        if copy_molecules[row][159] == 1:
            features.append('GE_T4_GE_T8_SUM')
        if copy_molecules[row][160] == 1:
            features.append('GE_T4_GE_T8_PRO')
        if copy_molecules[row][161] == 1:
            features.append('GE_T4_GE_T8_SUB')
        if copy_molecules[row][162] == 1:
            features.append('GE_T5')
        if copy_molecules[row][163] == 1:
            features.append('GE_T5_GE_T6_SUM')
        if copy_molecules[row][164] == 1:
            features.append('GE_T5_GE_T6_PRO')
        if copy_molecules[row][165] == 1:
            features.append('GE_T5_GE_T6_SUB')
        if copy_molecules[row][166] == 1:
            features.append('GE_T5_GE_T7_SUM')
        if copy_molecules[row][167] == 1:
            features.append('GE_T5_GE_T7_PRO')
        if copy_molecules[row][168] == 1:
            features.append('GE_T5_GE_T7_SUB')
        if copy_molecules[row][169] == 1:
            features.append('GE_T5_GE_T8_SUM')
        if copy_molecules[row][170] == 1:
            features.append('GE_T5_GE_T8_PRO')
        if copy_molecules[row][171] == 1:
            features.append('GE_T5_GE_T8_SUB')
        if copy_molecules[row][172] == 1:
            features.append('GE_T6')
        if copy_molecules[row][173] == 1:
            features.append('GE_T6_GE_T7_SUM')
        if copy_molecules[row][174] == 1:
            features.append('GE_T6_GE_T7_PRO')
        if copy_molecules[row][175] == 1:
            features.append('GE_T6_GE_T7_SUB')
        if copy_molecules[row][176] == 1:
            features.append('GE_T6_GE_T8_SUM')
        if copy_molecules[row][177] == 1:
            features.append('GE_T6_GE_T8_PRO')
        if copy_molecules[row][178] == 1:
            features.append('GE_T6_GE_T8_SUB')
        if copy_molecules[row][179] == 1:
            features.append('GE_T7')
        if copy_molecules[row][180] == 1:
            features.append('GE_T7_GE_T8_SUM')
        if copy_molecules[row][181] == 1:
            features.append('GE_T7_GE_T8_PRO')
        if copy_molecules[row][182] == 1:
            features.append('GE_T7_GE_T8_SUB')
        if copy_molecules[row][183] == 1:
            features.append('GE_T8')

        similar = 0;
        for j in range(184):
            if copy_molecules[row][j] == 0:
                similar = similar + 1
        c = c + 1
        print('{}. Current Molecule: {}    Current Features: {}'.format(c,copy_molecules[row], features))

        # read dataset
        n = TotalMolecule
        X = df[features]
        y = df['Essentiality']
        smenn = SMOTEENN()
        X_smote, y_smote = smenn.fit_resample(X, y)

        print(y_smote.value_counts())
        cal_accuracy(self, X_smote, y_smote)

    CRI_iterations(self)


def cal_accuracy(self, X, y):
    global MinPE, MinStruct, NumHit, PE, KE, X_smote, y_smote

    # Remove Comment from the line 1089 to execute the program using XGBoost Classifier.
    # Make comments from lines 1103 to 1115.
    #clf = XGBClassifier(max_depth = 3,  scale_pos_weight=4)
    # Remove Comment from the line 1092 to execute the program using XGBoost Classifier.
    # Make comments from lines 1103 to 1115.
    #clf = lgb.LGBMClassifier()
    # Remove Comment from the line 1095 to execute the program using Random Forest Classifier.
    # Make comments from lines 1103 to 1115.
    clf = RandomForestClassifier(n_estimators=100)

    #Remove comments from line 1099 to execute the program using XGBoost, LightGBM, or Random Forest classifier.

    scores = cross_val_score(estimator=clf, X=X, y=y, cv=10, scoring='accuracy')

    # Remove comments from lines 1103 to 1115 to execute the program using the ensemble method.
    #Make comments lines from 1089 to 1099.
    #estimators = []
    #estimators.clear()
    #clf = XGBClassifier(max_depth=4, scale_pos_weight=3)
    #estimators.append(('XGBoost', clf))

    #clf2 = RandomForestClassifier(n_estimators=100)
    #estimators.append(('RandomForest', clf2))

    #clf3 = lgb.LGBMClassifier()
    #estimators.append(('LightGBM', clf3))

    #ensemble = VotingClassifier(estimators)
    #scores = cross_val_score(estimator=ensemble, X=X_smote, y=y_smote, cv=10, scoring='accuracy')

    score = round(scores.mean() * 100, 4)
    print(score)

    PE.append(score)
    KE.append(100)
    NumHit.append(0)
    if MinPE < score:
        MinPE = score
        MinStruct.clear()
        MinStruct = features.copy()
    print('MinPE: {} MinStruct: {}'.format(MinPE, MinStruct))
    j = 0


def CRI_iterations(self):
    global molecule, PE, NumHit, buffer, MinStruct, MinPE, MinHit, buffer, TotalMolecule, T_NumHit, T_MinHit, copy_molecules
    T_MinHit = 0
    terminate = 0
    P_MinPE = 0
    MinPE = 0
    count_decom = 0
    count_inter = 0
    count_onwall = 0
    count_synt = 0
    iterate = 1

    while terminate < 1000:
        # Condition For CRO Termination
        print('Iterate no: {}'.format(iterate))
        iterate  = iterate + 1
        t = random.uniform(0, 1)
        # Decides whether to perform unimoleculat or inter molecular reaction
        if t > MoleColl:
            alpha = random.randint(0, 40)
            # Randomly Select a Molecule for Uni Molecular Reaction
            child = random.randint(0, TotalMolecule - 1)

            if (T_NumHit - T_MinHit) > alpha:
                # New Child Molecule 1
                print('                         Decomposition')
                count_decom = count_decom + 1
                copy_molecules = molecule.copy()


                copy_molecules[child][92] = 1
                copy_molecules[child][93] = 1
                copy_molecules[child][94] = 1
                copy_molecules[child][95] = 1
                copy_molecules[child][96] = 1
                copy_molecules[child][97] = 1
                copy_molecules[child][98] = 1
                copy_molecules[child][99] = 1
                copy_molecules[child][100] = 1
                copy_molecules[child][101] = 1
                copy_molecules[child][102] = 1
                copy_molecules[child][103] = 1
                copy_molecules[child][104] = 1
                copy_molecules[child][105] = 1
                copy_molecules[child][106] = 1
                copy_molecules[child][107] = 1
                copy_molecules[child][108] = 1
                copy_molecules[child][109] = 1
                copy_molecules[child][110] = 1
                copy_molecules[child][111] = 1
                copy_molecules[child][112] = 1
                copy_molecules[child][113] = 1
                copy_molecules[child][114] = 1
                copy_molecules[child][115] = 1
                copy_molecules[child][116] = 1
                copy_molecules[child][117] = 1
                copy_molecules[child][118] = 1
                copy_molecules[child][119] = 1
                copy_molecules[child][120] = 1
                copy_molecules[child][121] = 1
                copy_molecules[child][122] = 1
                copy_molecules[child][123] = 1
                copy_molecules[child][124] = 1
                copy_molecules[child][125] = 1
                copy_molecules[child][126] = 1
                copy_molecules[child][127] = 1
                copy_molecules[child][128] = 1
                copy_molecules[child][129] = 1
                copy_molecules[child][130] = 1
                copy_molecules[child][131] = 1
                copy_molecules[child][132] = 1
                copy_molecules[child][133] = 1
                copy_molecules[child][134] = 1
                copy_molecules[child][135] = 1
                copy_molecules[child][136] = 1
                copy_molecules[child][137] = 1
                copy_molecules[child][138] = 1
                copy_molecules[child][139] = 1
                copy_molecules[child][140] = 1
                copy_molecules[child][141] = 1
                copy_molecules[child][142] = 1
                copy_molecules[child][143] = 1
                copy_molecules[child][144] = 1
                copy_molecules[child][145] = 1
                copy_molecules[child][146] = 1
                copy_molecules[child][147] = 1
                copy_molecules[child][148] = 1
                copy_molecules[child][149] = 1
                copy_molecules[child][150] = 1
                copy_molecules[child][151] = 1
                copy_molecules[child][152] = 1
                copy_molecules[child][153] = 1
                copy_molecules[child][154] = 1
                copy_molecules[child][155] = 1
                copy_molecules[child][156] = 1
                copy_molecules[child][157] = 1
                copy_molecules[child][158] = 1
                copy_molecules[child][159] = 1
                copy_molecules[child][160] = 1
                copy_molecules[child][161] = 1
                copy_molecules[child][162] = 1
                copy_molecules[child][163] = 1
                copy_molecules[child][164] = 1
                copy_molecules[child][165] = 1
                copy_molecules[child][166] = 1
                copy_molecules[child][167] = 1
                copy_molecules[child][168] = 1
                copy_molecules[child][169] = 1
                copy_molecules[child][170] = 1
                copy_molecules[child][171] = 1
                copy_molecules[child][172] = 1
                copy_molecules[child][173] = 1
                copy_molecules[child][174] = 1
                copy_molecules[child][175] = 1
                copy_molecules[child][176] = 1
                copy_molecules[child][177] = 1
                copy_molecules[child][178] = 1
                copy_molecules[child][179] = 1
                copy_molecules[child][180] = 1
                copy_molecules[child][181] = 1
                copy_molecules[child][182] = 1
                copy_molecules[child][183] = 1


                # New Child Molecule 2
                copy_molecules2 = molecule.copy()
                copy_molecules2[child][0] = 1
                copy_molecules2[child][1] = 1
                copy_molecules2[child][2] = 1
                copy_molecules2[child][3] = 1
                copy_molecules2[child][4] = 1
                copy_molecules2[child][5] = 1
                copy_molecules2[child][6] = 1
                copy_molecules2[child][7] = 1
                copy_molecules2[child][8] = 1
                copy_molecules2[child][9] = 1
                copy_molecules2[child][10] = 1
                copy_molecules2[child][11] = 1
                copy_molecules2[child][12] = 1
                copy_molecules2[child][13] = 1
                copy_molecules2[child][14] = 1
                copy_molecules2[child][15] = 1
                copy_molecules2[child][16] = 1
                copy_molecules2[child][17] = 1
                copy_molecules2[child][18] = 1
                copy_molecules2[child][19] = 1
                copy_molecules2[child][20] = 1
                copy_molecules2[child][21] = 1
                copy_molecules2[child][22] = 1
                copy_molecules2[child][23] = 1
                copy_molecules2[child][24] = 1
                copy_molecules2[child][25] = 1
                copy_molecules2[child][26] = 1
                copy_molecules2[child][27] = 1
                copy_molecules2[child][28] = 1
                copy_molecules2[child][29] = 1
                copy_molecules2[child][30] = 1
                copy_molecules2[child][31] = 1
                copy_molecules2[child][32] = 1
                copy_molecules2[child][33] = 1
                copy_molecules2[child][34] = 1
                copy_molecules2[child][35] = 1
                copy_molecules2[child][36] = 1
                copy_molecules2[child][37] = 1
                copy_molecules2[child][38] = 1
                copy_molecules2[child][39] = 1
                copy_molecules2[child][40] = 1
                copy_molecules2[child][41] = 1
                copy_molecules2[child][42] = 1
                copy_molecules2[child][43] = 1
                copy_molecules2[child][44] = 1
                copy_molecules2[child][45] = 1
                copy_molecules2[child][46] = 1
                copy_molecules2[child][47] = 1
                copy_molecules2[child][48] = 1
                copy_molecules2[child][49] = 1
                copy_molecules2[child][50] = 1
                copy_molecules2[child][51] = 1
                copy_molecules2[child][52] = 1
                copy_molecules2[child][53] = 1
                copy_molecules2[child][54] = 1
                copy_molecules2[child][55] = 1
                copy_molecules2[child][56] = 1
                copy_molecules2[child][57] = 1
                copy_molecules2[child][58] = 1
                copy_molecules2[child][59] = 1
                copy_molecules2[child][60] = 1
                copy_molecules2[child][61] = 1
                copy_molecules2[child][62] = 1
                copy_molecules2[child][63] = 1
                copy_molecules2[child][64] = 1
                copy_molecules2[child][65] = 1
                copy_molecules2[child][66] = 1
                copy_molecules2[child][67] = 1
                copy_molecules2[child][68] = 1
                copy_molecules2[child][69] = 1
                copy_molecules2[child][70] = 1
                copy_molecules2[child][71] = 1
                copy_molecules2[child][72] = 1
                copy_molecules2[child][73] = 1
                copy_molecules2[child][74] = 1
                copy_molecules2[child][75] = 1
                copy_molecules2[child][76] = 1
                copy_molecules2[child][77] = 1
                copy_molecules2[child][78] = 1
                copy_molecules2[child][79] = 1
                copy_molecules2[child][80] = 1
                copy_molecules2[child][81] = 1
                copy_molecules2[child][82] = 1
                copy_molecules2[child][83] = 1
                copy_molecules2[child][84] = 1
                copy_molecules2[child][85] = 1
                copy_molecules2[child][86] = 1
                copy_molecules2[child][87] = 1
                copy_molecules2[child][88] = 1
                copy_molecules2[child][89] = 1
                copy_molecules2[child][90] = 1
                copy_molecules2[child][91] = 1

                # Calculate PE for Child Molecules
                print('Child1: {} Child2: {}'.format(copy_molecules[child], copy_molecules2[child]))
                acc_result_child_1 = cal_child_accuracy(self, copy_molecules, child)
                acc_result_child_2 = cal_child_accuracy(self, copy_molecules2, child)
                # PE for parent molecule
                acc_result_parent = PE[child]
                sigma_1 = random.uniform(0, 1)
                sigma_2 = random.uniform(0, 1)

                if (acc_result_parent + KE[child] + sigma_1 * sigma_2 * buffer) > (acc_result_child_1 + acc_result_child_2):
                    Edec = PE[child] + KE[child] + sigma_1 * sigma_2 * buffer - (acc_result_child_1 + acc_result_child_2)
                    sigma_3 = random.uniform(0, 1)
                    KE1 = Edec * sigma_3
                    KE2 = Edec * (1 - sigma_3)
                    buffer = buffer * (1 - sigma_1 * sigma_2)

                    KE[child] = KE1
                    PE[child] = acc_result_child_1
                    molecule[child] = copy_molecules[child].copy()
                    NumHit[child] = 0
                    KE.append(KE2)
                    PE.append(acc_result_child_2)
                    molecule.append(copy_molecules2[child])
                    NumHit.append(0)
                    TotalMolecule = TotalMolecule + 1
                    # New Optimal Features Found and Assign
                    if acc_result_child_1 > MinPE:
                        MinPE = acc_result_child_1
                        MinStruct = copy_molecules[child].copy()
                    elif acc_result_child_2 > MinPE:
                        MinPE = acc_result_child_2
                        MinStruct = copy_molecules2[child].copy()
                    NumHit[child] = 0
                    T_MinHit = T_NumHit
                T_NumHit = T_NumHit + 1

            # on-wall ineffective Collision
            else:
                print('                         On Wall Ineffective Collision')
                count_onwall = count_onwall + 1
                T_NumHit = T_NumHit + 1
                # Random number for molecule selection
                copy_molecules.clear()
                copy_molecules = molecule.copy()
                rand_position = random.randint(0, 183)

                if copy_molecules[child][rand_position] == 0:
                    copy_molecules[child][rand_position] = 1
                else:
                    copy_molecules[child][rand_position] = 0
                # Repair Function Reject zero or one feature
                similar = 0
                for j in range(184):
                    # for j in range(7):
                    if copy_molecules[child][j] == 0:
                        similar = similar + 1
                if similar >= 183:
                    continue
                print('Child: {} '.format(copy_molecules[child]))
                acc_result_child = cal_child_accuracy(self, copy_molecules, child)
                acc_result_parent = PE[child]
                if acc_result_child > acc_result_parent + KE[child]:
                    alpha1 = random.uniform(KELossRate, 1)
                    new_KE = (acc_result_parent - acc_result_child + KE[child]) * alpha1
                    buffer = buffer + (acc_result_parent - acc_result_child + KE[child]) * (1 - alpha1)
                    PE[child] = acc_result_child
                    KE[child] = new_KE
                    NumHit[child] = NumHit[child] + 1
                    molecule[child][rand_position] = copy_molecules[child][rand_position]

                    if MinPE < acc_result_child:
                        MinPE = acc_result_child
                        MinStruct = copy_molecules[child].copy()
                        MinHit = NumHit[child]
                        T_MinHit = T_NumHit

                # inter-molecular reaction
        else:
            random_child1 = -1
            random_child2 = -1
            # Select Two Child Molecules Number
            while random_child1 == random_child2:
                random_child1 = random.randint(0, TotalMolecule - 1)
                random_child2 = random.randint(0, TotalMolecule - 1)
            copy_molecules = molecule.copy()
            copy_molecules2 = molecule.copy()
            # Synthesis reaction condition
            if KE[random_child1] <= beta:
                print('                         Synthesis')
                count_synt = count_synt + 1

                copy_molecules[random_child1][92] = molecule[random_child2][92]
                copy_molecules[random_child1][93] = molecule[random_child2][93]
                copy_molecules[random_child1][94] = molecule[random_child2][94]
                copy_molecules[random_child1][95] = molecule[random_child2][95]
                copy_molecules[random_child1][96] = molecule[random_child2][96]
                copy_molecules[random_child1][97] = molecule[random_child2][97]
                copy_molecules[random_child1][98] = molecule[random_child2][98]
                copy_molecules[random_child1][99] = molecule[random_child2][99]
                copy_molecules[random_child1][100] = molecule[random_child2][100]
                copy_molecules[random_child1][101] = molecule[random_child2][101]
                copy_molecules[random_child1][102] = molecule[random_child2][102]
                copy_molecules[random_child1][103] = molecule[random_child2][103]
                copy_molecules[random_child1][104] = molecule[random_child2][104]
                copy_molecules[random_child1][105] = molecule[random_child2][105]
                copy_molecules[random_child1][106] = molecule[random_child2][106]
                copy_molecules[random_child1][107] = molecule[random_child2][107]
                copy_molecules[random_child1][108] = molecule[random_child2][108]
                copy_molecules[random_child1][109] = molecule[random_child2][109]
                copy_molecules[random_child1][110] = molecule[random_child2][110]
                copy_molecules[random_child1][111] = molecule[random_child2][111]
                copy_molecules[random_child1][112] = molecule[random_child2][112]
                copy_molecules[random_child1][113] = molecule[random_child2][113]
                copy_molecules[random_child1][114] = molecule[random_child2][114]
                copy_molecules[random_child1][115] = molecule[random_child2][115]
                copy_molecules[random_child1][116] = molecule[random_child2][116]
                copy_molecules[random_child1][117] = molecule[random_child2][117]
                copy_molecules[random_child1][118] = molecule[random_child2][118]
                copy_molecules[random_child1][119] = molecule[random_child2][119]
                copy_molecules[random_child1][120] = molecule[random_child2][120]
                copy_molecules[random_child1][121] = molecule[random_child2][121]
                copy_molecules[random_child1][122] = molecule[random_child2][122]
                copy_molecules[random_child1][123] = molecule[random_child2][123]
                copy_molecules[random_child1][124] = molecule[random_child2][124]
                copy_molecules[random_child1][125] = molecule[random_child2][125]
                copy_molecules[random_child1][126] = molecule[random_child2][126]
                copy_molecules[random_child1][127] = molecule[random_child2][127]
                copy_molecules[random_child1][128] = molecule[random_child2][128]
                copy_molecules[random_child1][129] = molecule[random_child2][129]
                copy_molecules[random_child1][130] = molecule[random_child2][130]
                copy_molecules[random_child1][131] = molecule[random_child2][131]
                copy_molecules[random_child1][132] = molecule[random_child2][132]
                copy_molecules[random_child1][133] = molecule[random_child2][133]
                copy_molecules[random_child1][134] = molecule[random_child2][134]
                copy_molecules[random_child1][135] = molecule[random_child2][135]
                copy_molecules[random_child1][136] = molecule[random_child2][136]
                copy_molecules[random_child1][137] = molecule[random_child2][137]
                copy_molecules[random_child1][138] = molecule[random_child2][138]
                copy_molecules[random_child1][139] = molecule[random_child2][139]
                copy_molecules[random_child1][140] = molecule[random_child2][140]
                copy_molecules[random_child1][141] = molecule[random_child2][141]
                copy_molecules[random_child1][142] = molecule[random_child2][142]
                copy_molecules[random_child1][143] = molecule[random_child2][143]
                copy_molecules[random_child1][144] = molecule[random_child2][144]
                copy_molecules[random_child1][145] = molecule[random_child2][145]
                copy_molecules[random_child1][146] = molecule[random_child2][146]
                copy_molecules[random_child1][147] = molecule[random_child2][147]
                copy_molecules[random_child1][148] = molecule[random_child2][148]
                copy_molecules[random_child1][149] = molecule[random_child2][149]
                copy_molecules[random_child1][150] = molecule[random_child2][150]
                copy_molecules[random_child1][151] = molecule[random_child2][151]
                copy_molecules[random_child1][152] = molecule[random_child2][152]
                copy_molecules[random_child1][153] = molecule[random_child2][153]
                copy_molecules[random_child1][154] = molecule[random_child2][154]
                copy_molecules[random_child1][155] = molecule[random_child2][155]
                copy_molecules[random_child1][156] = molecule[random_child2][156]
                copy_molecules[random_child1][157] = molecule[random_child2][157]
                copy_molecules[random_child1][158] = molecule[random_child2][158]
                copy_molecules[random_child1][159] = molecule[random_child2][159]
                copy_molecules[random_child1][160] = molecule[random_child2][160]
                copy_molecules[random_child1][161] = molecule[random_child2][161]
                copy_molecules[random_child1][162] = molecule[random_child2][162]
                copy_molecules[random_child1][163] = molecule[random_child2][163]
                copy_molecules[random_child1][164] = molecule[random_child2][164]
                copy_molecules[random_child1][165] = molecule[random_child2][165]
                copy_molecules[random_child1][166] = molecule[random_child2][166]
                copy_molecules[random_child1][167] = molecule[random_child2][167]
                copy_molecules[random_child1][168] = molecule[random_child2][168]
                copy_molecules[random_child1][169] = molecule[random_child2][169]
                copy_molecules[random_child1][170] = molecule[random_child2][170]
                copy_molecules[random_child1][171] = molecule[random_child2][171]
                copy_molecules[random_child1][172] = molecule[random_child2][172]
                copy_molecules[random_child1][173] = molecule[random_child2][173]
                copy_molecules[random_child1][174] = molecule[random_child2][174]
                copy_molecules[random_child1][175] = molecule[random_child2][175]
                copy_molecules[random_child1][176] = molecule[random_child2][176]
                copy_molecules[random_child1][177] = molecule[random_child2][177]
                copy_molecules[random_child1][178] = molecule[random_child2][178]
                copy_molecules[random_child1][179] = molecule[random_child2][179]
                copy_molecules[random_child1][180] = molecule[random_child2][180]
                copy_molecules[random_child1][181] = molecule[random_child2][181]
                copy_molecules[random_child1][182] = molecule[random_child2][182]
                copy_molecules[random_child1][183] = molecule[random_child2][183]

                similar = 0
                for j in range(184):
                    if copy_molecules[random_child1][j] == 0:
                        similar = similar + 1
                if similar >= 183:
                    print(copy_molecules[random_child1])
                    continue
                similar = 0
                for j in range(184):
                    if copy_molecules[random_child2][j] == 0:
                        similar = similar + 1

                print('Child: {} '.format(copy_molecules[random_child1]))

                acc_result_child_1 = cal_child_accuracy(self, copy_molecules, random_child1)

                acc_result_parent1 = PE[random_child1]
                acc_result_parent2 = PE[random_child2]
                KE_result_parent1 = KE[random_child1]
                KE_result_parent2 = KE[random_child2]
                if (acc_result_parent1 + acc_result_parent2 + KE_result_parent1 + KE_result_parent2) > acc_result_child_1:
                    KE[random_child1] = (acc_result_parent1 + acc_result_parent2 + KE_result_parent1 + KE_result_parent2) - acc_result_child_1
                    PE[random_child1] = acc_result_child_1
                    NumHit[random_child1] = 0
                    if MinPE < acc_result_child_1:
                        MinPE = acc_result_child_1
                        MinStruct = copy_molecules[random_child1].copy()
                    index = molecule.index(copy_molecules[random_child2])
                    del molecule[index]
                    del NumHit[index]
                    del PE[index]
                    del KE[index]
                    T_MinHit = T_NumHit
                    TotalMolecule = TotalMolecule - 1

                else:
                    NumHit[random_child1] = NumHit[random_child1] + 1
                    NumHit[random_child2] = NumHit[random_child2] + 1
                T_NumHit = T_NumHit + 1

            else:
                print('                         Inter Molecule')
                count_inter = count_inter + 1
                random_position = random.randint(0, 183)
                if copy_molecules[random_child1][random_position] == 1:
                    copy_molecules[random_child1][random_position] = 0
                else:
                    copy_molecules[random_child1][random_position] = 1

                similar = 0
                for j in range(184):
                    if copy_molecules[random_child1][j] == 0:
                        similar = similar + 1
                if similar >= 183:
                    print('Ignore synthesis for 1 or 0 features: ')
                    print(copy_molecules[random_child1])
                    continue
                similar = 0
                for j in range(184):
                    if copy_molecules[random_child2][j] == 0:
                        similar = similar + 1
                if similar >= 183:
                    print('Ignore synthesis for 1 or 0 features: ')
                    print(copy_molecules[random_child2])
                    continue
                print('Child1: {} Child2: {}'.format(copy_molecules[random_child1], copy_molecules[random_child2]))
                acc_result_child_1 = cal_child_accuracy(self, copy_molecules, random_child1)
                acc_result_child_2 = cal_child_accuracy(self, copy_molecules2, random_child2)
                NumHit[random_child1] = NumHit[random_child1] + 1
                NumHit[random_child2] = NumHit[random_child2] + 1
                T_NumHit = T_NumHit + 1
                acc_result_parent1 = PE[random_child1]
                acc_result_parent2 = PE[random_child2]
                KE_result_parent1 = KE[random_child1]
                KE_result_parent2 = KE[random_child2]
                E_inter = (acc_result_parent1 + acc_result_parent2 + KE_result_parent1 + KE_result_parent2) - (
                        acc_result_child_1 + acc_result_child_2)
                if E_inter > 0:
                    Sigma_4 = random.uniform(0, 1)
                    KE1 = E_inter * Sigma_4
                    KE2 = E_inter * (1 - Sigma_4)
                    molecule[random_child1] = copy_molecules[random_child1].copy()
                    molecule[random_child2] = copy_molecules2[random_child2].copy()
                    PE[random_child1] = acc_result_child_1
                    PE[random_child2] = acc_result_child_2
                    KE[random_child1] = KE1
                    KE[random_child2] = KE2
                    NumHit[random_child1] = NumHit[random_child1] + 1
                    NumHit[random_child2] = NumHit[random_child2] + 1

                    if acc_result_child_1 > MinPE and acc_result_child_1 > acc_result_child_2:
                        MinPE = acc_result_child_1
                        MinStruct = molecule[random_child1].copy()
                        MinHit = NumHit[random_child1]
                        T_MinHit = T_NumHit
                    if acc_result_child_2 > MinPE and acc_result_child_1 < acc_result_child_2:
                        MinPE = acc_result_child_2
                        MinStruct = molecule[random_child2].copy()
                        MinHit = NumHit[random_child2]
                        T_MinHit = T_NumHit

                # print('Inter Molecular Ineffective End')
        print('Optimal Accuracy:  {} Overall Accuracy: {}  Optimal Features: {} Terminate: {}'.format(MinPE, P_MinPE,
                                                                                                      MinStruct,
                                                                                                      terminate))
        print(MinStruct)
        if MinPE > P_MinPE:
            terminate = 0
            P_MinPE = MinPE
        else:
            terminate = terminate + 1
    print('Decomposition: {} Synthesis: {} On-Wall: {} Inter-Molecular: {}'.format(count_decom,count_synt,count_onwall,count_inter))


def cal_child_accuracy(self, child_Mol, child):
    global features, X, y
    features.clear()

    if child_Mol[child][0] == 1:
        features.append('SG')
    if child_Mol[child][1] == 1:
        features.append('DG')
    if child_Mol[child][2] == 1:
        features.append('SG_DG_SUM')
    if child_Mol[child][3] == 1:
        features.append('SG_DG_PRO')
    if child_Mol[child][4] == 1:
        features.append('SG_DG_SUB')
    if child_Mol[child][5] == 1:
        features.append('EV')
    if copy_molecules[child][6] == 1:
        features.append('SG_EV_SUM')
    if child_Mol[child][7] == 1:
        features.append('SG_EV_PRO')
    if child_Mol[child][8] == 1:
        features.append('SG_EV_SUB')
    if child_Mol[child][9] == 1:
        features.append('DG_EV_SUM')
    if child_Mol[child][10] == 1:
        features.append('DG_EV_PRO')
    if child_Mol[child][11] == 1:
        features.append('DG_EV_SUB')
    if child_Mol[child][12] == 1:
        features.append('INFO')
    if child_Mol[child][13] == 1:
        features.append('SG_INFO_SUM')
    if child_Mol[child][14] == 1:
        features.append('SG_INFO_PRO')
    if child_Mol[child][15] == 1:
        features.append('SG_INFO_SUB')
    if child_Mol[child][16] == 1:
        features.append('DG_INFO_SUM')
    if child_Mol[child][17] == 1:
        features.append('DG_INFO_PRO')
    if child_Mol[child][18] == 1:
        features.append('DG_INFO_SUB')
    if child_Mol[child][19] == 1:
        features.append('EV_INFO_SUM')
    if child_Mol[child][20] == 1:
        features.append('EV_INFO_PRO')
    if child_Mol[child][21] == 1:
        features.append('EV_INFO_SUB')
    if child_Mol[child][22] == 1:
        features.append('LAC')
    if child_Mol[child][23] == 1:
        features.append('SG_LAC_SUM')
    if child_Mol[child][24] == 1:
        features.append('SG_LAC_PRO')
    if child_Mol[child][25] == 1:
        features.append('SG_LAC_SUB')
    if child_Mol[child][26] == 1:
        features.append('DG_LAC_SUM')
    if child_Mol[child][27] == 1:
        features.append('DG_LAC_PRO')
    if child_Mol[child][28] == 1:
        features.append('DG_LAC_SUB')
    if child_Mol[child][29] == 1:
        features.append('EV_LAC_SUM')
    if child_Mol[child][30] == 1:
        features.append('EV_LAC_PRO')
    if child_Mol[child][31] == 1:
        features.append('EV_LAC_SUB')
    if child_Mol[child][32] == 1:
        features.append('INFO_LAC_SUM')
    if child_Mol[child][33] == 1:
        features.append('INFO_LAC_PRO')
    if child_Mol[child][34] == 1:
        features.append('INFO_LAC_SUB')
    if child_Mol[child][35] == 1:
        features.append('BNESS')
    if child_Mol[child][36] == 1:
        features.append('SG_BNESS_SUM')
    if child_Mol[child][37] == 1:
        features.append('SG_BNESS_PRO')
    if child_Mol[child][38] == 1:
        features.append('SG_BNESS_SUB')
    if child_Mol[child][39] == 1:
        features.append('DG_BNESS_SUM')
    if child_Mol[child][40] == 1:
        features.append('DG_BNESS_PRO')
    if child_Mol[child][41] == 1:
        features.append('DG_BNESS_SUB')
    if child_Mol[child][42] == 1:
        features.append('EV_BNESS_SUM')
    if child_Mol[child][43] == 1:
        features.append('EV_BNESS_PRO')
    if child_Mol[child][44] == 1:
        features.append('EV_BNESS_SUB')
    if child_Mol[child][45] == 1:
        features.append('INFO_BNESS_SUM')
    if child_Mol[child][46] == 1:
        features.append('INFO_BNESS_PRO')
    if child_Mol[child][47] == 1:
        features.append('INFO_BNESS_SUB')
    if child_Mol[child][48] == 1:
        features.append('LAC_BNESS_SUM')
    if child_Mol[child][49] == 1:
        features.append('LAC_BNESS_PRO')
    if child_Mol[child][50] == 1:
        features.append('LAC_BNESS_SUB')
    if child_Mol[child][51] == 1:
        features.append('CNESS')
    if child_Mol[child][52] == 1:
        features.append('SG_CNESS_SUM')
    if child_Mol[child][53] == 1:
        features.append('SG_CNESS_PRO')
    if child_Mol[child][54] == 1:
        features.append('SG_CNESS_SUB')
    if child_Mol[child][55] == 1:
        features.append('DG_CNESS_SUM')
    if child_Mol[child][56] == 1:
        features.append('DG_CNESS_PRO')
    if child_Mol[child][57] == 1:
        features.append('DG_CNESS_SUB')
    if child_Mol[child][58] == 1:
        features.append('EV_CNESS_SUM')
    if child_Mol[child][59] == 1:
        features.append('EV_CNESS_PRO')
    if child_Mol[child][60] == 1:
        features.append('EV_CNESS_SUB')
    if child_Mol[child][61] == 1:
        features.append('INFO_CNESS_SUM')
    if child_Mol[child][62] == 1:
        features.append('INFO_CNESS_PRO')
    if child_Mol[child][63] == 1:
        features.append('INFO_CNESS_SUB')
    if child_Mol[child][64] == 1:
        features.append('LAC_CNESS_SUM')
    if child_Mol[child][65] == 1:
        features.append('LAC_CNESS_PRO')
    if child_Mol[child][66] == 1:
        features.append('LAC_CNESS_SUB')
    if child_Mol[child][67] == 1:
        features.append('BNESS_CNESS_SUM')
    if child_Mol[child][68] == 1:
        features.append('BNESS_CNESS_PRO')
    if child_Mol[child][69] == 1:
        features.append('BNESS_CNESS_SUB')
    if child_Mol[child][70] == 1:
        features.append('NET')
    if child_Mol[child][71] == 1:
        features.append('SG_NET_SUM')
    if child_Mol[child][72] == 1:
        features.append('SG_NET_PRO')
    if child_Mol[child][73] == 1:
        features.append('SG_NET_SUB')
    if child_Mol[child][74] == 1:
        features.append('DG_NET_SUM')
    if child_Mol[child][75] == 1:
        features.append('DG_NET_PRO')
    if child_Mol[child][76] == 1:
        features.append('DG_NET_SUB')
    if child_Mol[child][77] == 1:
        features.append('EV_NET_SUM')
    if child_Mol[child][78] == 1:
        features.append('EV_NET_PRO')
    if child_Mol[child][79] == 1:
        features.append('EV_NET_SUB')
    if child_Mol[child][80] == 1:
        features.append('INFO_NET_SUM')
    if child_Mol[child][81] == 1:
        features.append('INFO_NET_PRO')
    if child_Mol[child][82] == 1:
        features.append('INFO_NET_SUB')
    if child_Mol[child][83] == 1:
        features.append('LAC_NET_SUM')
    if child_Mol[child][84] == 1:
        features.append('LAC_NET_PRO')
    if child_Mol[child][85] == 1:
        features.append('LAC_NET_SUB')
    if child_Mol[child][86] == 1:
        features.append('BNESS_NET_SUM')
    if child_Mol[child][87] == 1:
        features.append('BNESS_NET_PRO')
    if child_Mol[child][88] == 1:
        features.append('BNESS_NET_SUB')
    if child_Mol[child][89] == 1:
        features.append('CNESS_NET_SUM')
    if child_Mol[child][90] == 1:
        features.append('CNESS_NET_PRO')
    if child_Mol[child][91] == 1:
        features.append('CNESS_NET_SUB')
    if child_Mol[child][92] == 1:
        features.append('GE_T1')
    if child_Mol[child][93] == 1:
        features.append('GE_T1_GE_T2_SUM')
    if child_Mol[child][94] == 1:
        features.append('GE_T1_GE_T2_PRO')
    if child_Mol[child][95] == 1:
        features.append('GE_T1_GE_T2_SUB')
    if child_Mol[child][96] == 1:
        features.append('GE_T1_GE_T3_SUM')
    if child_Mol[child][97] == 1:
        features.append('GE_T1_GE_T3_PRO')
    if child_Mol[child][98] == 1:
        features.append('GE_T1_GE_T3_SUB')
    if child_Mol[child][99] == 1:
        features.append('GE_T1_GE_T4_SUM')
    if child_Mol[child][100] == 1:
        features.append('GE_T1_GE_T4_PRO')
    if child_Mol[child][101] == 1:
        features.append('GE_T1_GE_T4_SUB')
    if child_Mol[child][102] == 1:
        features.append('GE_T1_GE_T5_SUM')
    if child_Mol[child][103] == 1:
        features.append('GE_T1_GE_T5_PRO')
    if child_Mol[child][104] == 1:
        features.append('GE_T1_GE_T5_SUB')
    if child_Mol[child][105] == 1:
        features.append('GE_T1_GE_T6_SUM')
    if child_Mol[child][106] == 1:
        features.append('GE_T1_GE_T6_PRO')
    if child_Mol[child][107] == 1:
        features.append('GE_T1_GE_T6_SUB')
    if child_Mol[child][108] == 1:
        features.append('GE_T1_GE_T7_SUM')
    if child_Mol[child][109] == 1:
        features.append('GE_T1_GE_T7_PRO')
    if child_Mol[child][110] == 1:
        features.append('GE_T1_GE_T7_SUB')
    if child_Mol[child][111] == 1:
        features.append('GE_T1_GE_T8_SUM')
    if child_Mol[child][112] == 1:
        features.append('GE_T1_GE_T8_PRO')
    if child_Mol[child][113] == 1:
        features.append('GE_T1_GE_T8_SUB')
    if child_Mol[child][114] == 1:
        features.append('GE_T2')
    if child_Mol[child][115] == 1:
        features.append('GE_T2_GE_T3_SUM')
    if child_Mol[child][116] == 1:
        features.append('GE_T2_GE_T3_PRO')
    if child_Mol[child][117] == 1:
        features.append('GE_T2_GE_T3_SUB')
    if child_Mol[child][118] == 1:
        features.append('GE_T2_GE_T4_SUM')
    if child_Mol[child][119] == 1:
        features.append('GE_T2_GE_T4_PRO')
    if child_Mol[child][120] == 1:
        features.append('GE_T2_GE_T4_SUB')
    if child_Mol[child][121] == 1:
        features.append('GE_T2_GE_T5_SUM')
    if child_Mol[child][122] == 1:
        features.append('GE_T2_GE_T5_PRO')
    if child_Mol[child][123] == 1:
        features.append('GE_T2_GE_T5_SUB')
    if child_Mol[child][124] == 1:
        features.append('GE_T2_GE_T6_SUM')
    if child_Mol[child][125] == 1:
        features.append('GE_T2_GE_T6_PRO')
    if child_Mol[child][126] == 1:
        features.append('GE_T2_GE_T6_SUB')
    if child_Mol[child][127] == 1:
        features.append('GE_T2_GE_T7_SUM')
    if child_Mol[child][128] == 1:
        features.append('GE_T2_GE_T7_PRO')
    if child_Mol[child][129] == 1:
        features.append('GE_T2_GE_T7_SUB')
    if child_Mol[child][130] == 1:
        features.append('GE_T2_GE_T8_SUM')
    if child_Mol[child][131] == 1:
        features.append('GE_T2_GE_T8_PRO')
    if child_Mol[child][132] == 1:
        features.append('GE_T2_GE_T8_SUB')
    if child_Mol[child][133] == 1:
        features.append('GE_T3')
    if child_Mol[child][134] == 1:
        features.append('GE_T3_GE_T4_SUM')
    if child_Mol[child][135] == 1:
        features.append('GE_T3_GE_T4_PRO')
    if child_Mol[child][136] == 1:
        features.append('GE_T3_GE_T4_SUB')
    if child_Mol[child][137] == 1:
        features.append('GE_T3_GE_T5_SUM')
    if child_Mol[child][138] == 1:
        features.append('GE_T3_GE_T5_PRO')
    if child_Mol[child][139] == 1:
        features.append('GE_T3_GE_T5_SUB')
    if child_Mol[child][140] == 1:
        features.append('GE_T3_GE_T6_SUM')
    if child_Mol[child][141] == 1:
        features.append('GE_T3_GE_T6_PRO')
    if child_Mol[child][142] == 1:
        features.append('GE_T3_GE_T6_SUB')
    if child_Mol[child][143] == 1:
        features.append('GE_T3_GE_T7_SUM')
    if child_Mol[child][144] == 1:
        features.append('GE_T3_GE_T7_PRO')
    if child_Mol[child][145] == 1:
        features.append('GE_T3_GE_T7_SUB')
    if child_Mol[child][146] == 1:
        features.append('GE_T3_GE_T8_SUM')
    if child_Mol[child][147] == 1:
        features.append('GE_T3_GE_T8_PRO')
    if child_Mol[child][148] == 1:
        features.append('GE_T3_GE_T8_SUB')
    if child_Mol[child][149] == 1:
        features.append('GE_T4')
    if child_Mol[child][150] == 1:
        features.append('GE_T4_GE_T5_SUM')
    if child_Mol[child][151] == 1:
        features.append('GE_T4_GE_T5_PRO')
    if child_Mol[child][152] == 1:
        features.append('GE_T4_GE_T5_SUB')
    if child_Mol[child][153] == 1:
        features.append('GE_T4_GE_T6_SUM')
    if child_Mol[child][154] == 1:
        features.append('GE_T4_GE_T6_PRO')
    if child_Mol[child][155] == 1:
        features.append('GE_T4_GE_T6_SUB')
    if child_Mol[child][156] == 1:
        features.append('GE_T4_GE_T7_SUM')
    if child_Mol[child][157] == 1:
        features.append('GE_T4_GE_T7_PRO')
    if child_Mol[child][158] == 1:
        features.append('GE_T4_GE_T7_SUB')
    if child_Mol[child][159] == 1:
        features.append('GE_T4_GE_T8_SUM')
    if child_Mol[child][160] == 1:
        features.append('GE_T4_GE_T8_PRO')
    if child_Mol[child][161] == 1:
        features.append('GE_T4_GE_T8_SUB')
    if child_Mol[child][162] == 1:
        features.append('GE_T5')
    if child_Mol[child][163] == 1:
        features.append('GE_T5_GE_T6_SUM')
    if child_Mol[child][164] == 1:
        features.append('GE_T5_GE_T6_PRO')
    if child_Mol[child][165] == 1:
        features.append('GE_T5_GE_T6_SUB')
    if child_Mol[child][166] == 1:
        features.append('GE_T5_GE_T7_SUM')
    if child_Mol[child][167] == 1:
        features.append('GE_T5_GE_T7_PRO')
    if child_Mol[child][168] == 1:
        features.append('GE_T5_GE_T7_SUB')
    if child_Mol[child][169] == 1:
        features.append('GE_T5_GE_T8_SUM')
    if child_Mol[child][170] == 1:
        features.append('GE_T5_GE_T8_PRO')
    if child_Mol[child][171] == 1:
        features.append('GE_T5_GE_T8_SUB')
    if child_Mol[child][172] == 1:
        features.append('GE_T6')
    if child_Mol[child][173] == 1:
        features.append('GE_T6_GE_T7_SUM')
    if child_Mol[child][174] == 1:
        features.append('GE_T6_GE_T7_PRO')
    if child_Mol[child][175] == 1:
        features.append('GE_T6_GE_T7_SUB')
    if child_Mol[child][176] == 1:
        features.append('GE_T6_GE_T8_SUM')
    if child_Mol[child][177] == 1:
        features.append('GE_T6_GE_T8_PRO')
    if child_Mol[child][178] == 1:
        features.append('GE_T6_GE_T8_SUB')
    if child_Mol[child][179] == 1:
        features.append('GE_T7')
    if child_Mol[child][180] == 1:
        features.append('GE_T7_GE_T8_SUM')
    if child_Mol[child][181] == 1:
        features.append('GE_T7_GE_T8_PRO')
    if child_Mol[child][182] == 1:
        features.append('GE_T7_GE_T8_SUB')
    if child_Mol[child][183] == 1:
        features.append('GE_T8')
    X = df[features]
    y = df['Essentiality']
    # Remove comments from the line 2024 and 2025 to execute the program using imbalance dataset.
    # Make Comments from line 2028 and 2029.
    #X_smote = X
    #y_smote = y
    #Remove comments from the line 2028 to 2029 to execute the program using EMOTE-ENN balance dataset.
    #Make Comments from line 2024 and 2025.
    smenn = SMOTEENN()
    X_smote, y_smote = smenn.fit_resample(X, y)

    print(y_smote.value_counts())
    #Remove comment from line 2034 to execute the program using LightGBM.
    #Make comments from line 2050 to 2063.
    #clf = lgb.LGBMClassifier()

    #Remove comment from line 2038 to execute the program using XGBoost.
    #Make comments from line 2050 to 2063.
    #clf = XGBClassifier(max_depth = 3, scale_pos_weight=4)

    # Remove comment from line 2042 to execute the program using Random Forest.
    # Make comments from line 2050 to 2063.
    clf = RandomForestClassifier(n_estimators=100)

    # Remove comment from line 2046 and 2047 to execute the program using either LightGBM, XGBoost, or Random Forest.

    scores = cross_val_score(estimator=clf, X=X_smote, y=y_smote, cv=10, scoring='accuracy')
    predicted_label = cross_val_predict(estimator=clf, X=X_smote, y=y_smote, cv=10)
    # Remove comment from line 2050 to 2063 to execute the program using ensemble method.
    # Make comments from 2034 to 2047 lines.
    #estimators = []
    #estimators.clear()
    #clf = XGBClassifier(max_depth=4, scale_pos_weight=3)
    #estimators.append(('XGBoost', clf))

    #clf2 = RandomForestClassifier(n_estimators=100)
    #estimators.append(('RandomForest', clf2))

    #clf3 = lgb.LGBMClassifier()
    #estimators.append(('LightGBM', clf3))

    #ensemble = VotingClassifier(estimators)
    #scores = cross_val_score(estimator=ensemble, X=X_smote, y=y_smote, cv=10, scoring='accuracy')
    #predicted_label = cross_val_predict(estimator=clf, X=X_smote, y=y_smote, cv=10)

    score = round(scores.mean() * 100, 4)


    cm = np.array(confusion_matrix(y_smote, predicted_label))

    confusion = pd.DataFrame(cm, index=['NE', 'ES'], columns=['NE', 'ES'])

    CM = confusion_matrix(y_smote, predicted_label)
    print(confusion)
    print(score)
    return score

print('new done')

def main():
    call_CRO = CRO_class()
main()
