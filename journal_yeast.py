import random
import pandas
import numpy as np
from imblearn.combine import SMOTEENN
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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
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
from imblearn.over_sampling import SMOTE

class CRO_class:

    def __init__(self):
        global NumHit, T_NumHit, MinStruct, MinPE, MinHit, buffer, KELossRate, MoleColl, alpha, beta
        global TotalMolecule, PE, KE, MoleNumber, molecule, copy_molecules, copy_molecules2
        global X, Y, y, features, obj_func_gain,c
        # Create Random Molecules
        MoleNumber = random.randint(1, 100)
        # Create A List of Possible Total Number of Molecules
        molecule = [[0 for i in range(287)] for j in range(MoleNumber)]
        copy_molecules = [[0 for i in range(287)] for j in range(MoleNumber)]
        copy_molecules2 = [[0 for i in range(287)] for j in range(MoleNumber)]
        MinStruct = ['SL', 'DC', 'EC', 'IC', 'LAC', 'BC', 'CC', 'NC','ADA2','ADR1','AFT2','PC1','PC2','PC3','SL_DC_SUM','SL_DC_SUB','SL_DC_PRO','SL_EC_SUM','SL_EC_SUB','SL_EC_PRO','SL_IC_SUM','SL_IC_SUB','SL_IC_PRO','SL_LAC_SUM','SL_LAC_SUB','SL_LAC_PRO','SL_BC_SUM','SL_BC_SUB','SL_BC_PRO','SL_CC_SUM','SL_CC_SUB','SL_CC_PRO','SL_NC_SUM','SL_NC_SUB','SL_NC_PRO','SL_ADA2_SUM','SL_ADA2_SUB','SL_ADA2_PRO','SL_ADR1_SUM','SL_ADR1_SUB','SL_ADR1_PRO','SL_AFT2_SUM','SL_AFT2_SUB','SL_AFT2_PRO','SL_PC1_SUM','SL_PC1_SUB','SL_PC1_PRO','SL_PC2_SUM','SL_PC2_SUB','SL_PC2_PRO','SL_PC3_SUM','SL_PC3_SUB','SL_PC3_PRO','DC_EC_SUM','DC_EC_SUB','DC_EC_PRO','DC_IC_SUM','DC_IC_SUB','DC_IC_PRO','DC_LAC_SUM','DC_LAC_SUB','DC_LAC_PRO','DC_BC_SUM','DC_BC_SUB','DC_BC_PRO','DC_CC_SUM','DC_CC_SUB','DC_CC_PRO','DC_NC_SUM','DC_NC_SUB','DC_NC_PRO','DC_ADA2_SUM','DC_ADA2_SUB','DC_ADA2_PRO','DC_ADR1_SUM','DC_ADR1_SUB','DC_ADR1_PRO','DC_AFT2_SUM','DC_AFT2_SUB','DC_AFT2_PRO','DC_PC1_SUM','DC_PC1_SUB','DC_PC1_PRO','DC_PC2_SUM','DC_PC2_SUB','DC_PC2_PRO','DC_PC3_SUM','DC_PC3_SUB','DC_PC3_PRO','EC_IC_SUM','EC_IC_SUB','EC_IC_PRO','EC_LAC_SUM','EC_LAC_SUB','EC_LAC_PRO','EC_BC_SUM','EC_BC_SUB','EC_BC_PRO','EC_CC_SUM','EC_CC_SUB','EC_CC_PRO','EC_NC_SUM','EC_NC_SUB','EC_NC_PRO','EC_ADA2_SUM','EC_ADA2_SUB','EC_ADA2_PRO','EC_ADR1_SUM','EC_ADR1_SUB','EC_ADR1_PRO','EC_AFT2_SUM','EC_AFT2_SUB','EC_AFT2_PRO','EC_PC1_SUM','EC_PC1_SUB','EC_PC1_PRO','EC_PC2_SUM','EC_PC2_SUB','EC_PC2_PRO','EC_PC3_SUM','EC_PC3_SUB','EC_PC3_PRO','IC_LAC_SUM','IC_LAC_SUB','IC_LAC_PRO','IC_BC_SUM','IC_BC_SUB','IC_BC_PRO','IC_CC_SUM','IC_CC_SUB','IC_CC_PRO','IC_NC_SUM','IC_NC_SUB','IC_NC_PRO','IC_ADA2_SUM','IC_ADA2_SUB','IC_ADA2_PRO','IC_ADR1_SUM','IC_ADR1_SUB','IC_ADR1_PRO','IC_AFT2_SUM','IC_AFT2_SUB','IC_AFT2_PRO','IC_PC1_SUM','IC_PC1_SUB','IC_PC1_PRO','IC_PC2_SUM','IC_PC2_SUB','IC_PC2_PRO','IC_PC3_SUM','IC_PC3_SUB','IC_PC3_PRO','LAC_BC_SUM','LAC_BC_SUB','LAC_BC_PRO','LAC_CC_SUM','LAC_CC_SUB','LAC_CC_PRO','LAC_NC_SUM','LAC_NC_SUB','LAC_NC_PRO','LAC_ADA2_SUM','LAC_ADA2_SUB','LAC_ADA2_PRO','LAC_ADR1_SUM','LAC_ADR1_SUB','LAC_ADR1_PRO','LAC_AFT2_SUM','LAC_AFT2_SUB','LAC_AFT2_PRO','LAC_PC1_SUM','LAC_PC1_SUB','LAC_PC1_PRO','LAC_PC2_SUM','LAC_PC2_SUB','LAC_PC2_PRO','LAC_PC3_SUM','LAC_PC3_SUB','LAC_PC3_PRO','BC_CC_SUM','BC_CC_SUB','BC_CC_PRO','BC_NC_SUM','BC_NC_SUB','BC_NC_PRO','BC_ADA2_SUM','BC_ADA2_SUB','BC_ADA2_PRO','BC_ADR1_SUM','BC_ADR1_SUB','BC_ADR1_PRO','BC_AFT2_SUM','BC_AFT2_SUB','BC_AFT2_PRO','BC_PC1_SUM','BC_PC1_SUB','BC_PC1_PRO','BC_PC2_SUM','BC_PC2_SUB','BC_PC2_PRO','BC_PC3_SUM','BC_PC3_SUB','BC_PC3_PRO','CC_NC_SUM','CC_NC_SUB','CC_NC_PRO','CC_ADA2_SUM','CC_ADA2_SUB','CC_ADA2_PRO','CC_ADR1_SUM','CC_ADR1_SUB','CC_ADR1_PRO','CC_AFT2_SUM','CC_AFT2_SUB','CC_AFT2_PRO','CC_PC1_SUM','CC_PC1_SUB','CC_PC1_PRO','CC_PC2_SUM','CC_PC2_SUB','CC_PC2_PRO','CC_PC3_SUM','CC_PC3_SUB','CC_PC3_PRO','NC_ADA2_SUM','NC_ADA2_SUB','NC_ADA2_PRO','NC_ADR1_SUM','NC_ADR1_SUB','NC_ADR1_PRO','NC_AFT2_SUM','NC_AFT2_SUB','NC_AFT2_PRO','NC_PC1_SUM','NC_PC1_SUB','NC_PC1_PRO','NC_PC2_SUM','NC_PC2_SUB','NC_PC2_PRO','NC_PC3_SUM','NC_PC3_SUB','NC_PC3_PRO','ADA2_ADR1_SUM','ADA2_ADR1_SUB',	'ADA2_ADR1_PRO','ADA2_AFT2_SUM','ADA2_AFT2_SUB','ADA2_AFT2_PRO','ADA2_PC1_SUM','ADA2_PC1_SUB','ADA2_PC1_PRO','ADA2_PC2_SUM','ADA2_PC2_SUB','ADA2_PC2_PRO','ADA2_PC3_SUM','ADA2_PC3_SUB','ADA2_PC3_PRO','ADR1_AFT2_SUM','ADR1_AFT2_SUB','ADR1_AFT2_PRO','ADR1_PC1_SUM','ADR1_PC1_SUB','ADR1_PC1_PRO','ADR1_PC2_SUM','ADR1_PC2_SUB','ADR1_PC2_PRO','ADR1_PC3_SUM','ADR1_PC3_SUB','ADR1_PC3_PRO','AFT2_PC1_SUM','AFT2_PC1_SUB','AFT2_PC1_PRO','AFT2_PC2_SUM','AFT2_PC2_SUB','AFT2_PC2_PRO','AFT2_PC3_SUM','AFT2_PC3_SUB','AFT2_PC3_PRO','PC1_PC2_SUM','PC1_PC2_SUB','PC1_PC2_PRO','PC1_PC3_SUM','PC1_PC3_SUB','PC1_PC3_PRO','PC2_PC3_SUM','PC2_PC3_SUB','PC2_PC3_PRO']
        MinPE = 0
        MinHit = 0
        buffer = 100
        KELossRate = .2
        MoleColl = .6
        alpha = 70
        beta = 30
        TotalMolecule = 0
        T_NumHit = 0
        c = 0.00001
        obj_func_gain = 1.0
        Min_obj_func_val = 1.0
        # Assign Initial Activation Values to Molecules
        for row in range(MoleNumber):
            for j in range(287):
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
            for j in range(287):
                if molecule[row][j] == copy_molecules2[row2][j]:
                    similar = similar + 1
                if molecule[row][j] == 0:
                    all_zero = all_zero + 1

            if similar == 287 or all_zero == 287:
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
    global sl_X, dc_X, ec_X, ic_X, lac_X, bc_X, cc_X, nc_X, ada2_X, adr1_X, aft2_X, PC1_X, PC2_X, PC3_X
    # set datatype for features
    PE = [TotalMolecule]
    KE = [TotalMolecule]
    NumHit = [TotalMolecule]
    features = [287]
    copy_molecules.clear()
    copy_molecules = molecule.copy()

    df = pandas.read_csv("E:\yeast.csv", header=0)
    df.head()
    d = {'NE': 0, 'E': 1, 'E,NE': 1, 'NE.E': 0}
    df['P_Status'] = df['P_Status'].map(d)

    scaler = StandardScaler()

    sl_X = np.array(df["SL"]).reshape(-1, 1)
    df["SL"] = scaler.fit_transform(sl_X)

    dc_X = np.array(df["DC"]).reshape(-1, 1)
    df["DC"] = scaler.fit_transform(dc_X)

    ec_X = np.array(df["EC"]).reshape(-1, 1)
    df["EC"] = scaler.fit_transform(ec_X)

    ic_X = np.array(df["IC"]).reshape(-1, 1)
    df["IC"] = scaler.fit_transform(ic_X)

    lac_X = np.array(df["LAC"]).reshape(-1, 1)
    df["LAC"] = scaler.fit_transform(lac_X)

    bc_X = np.array(df["BC"]).reshape(-1, 1)
    df["BC"] = scaler.fit_transform(bc_X)

    cc_X = np.array(df["CC"]).reshape(-1, 1)
    df["CC"] = scaler.fit_transform(cc_X)

    nc_X = np.array(df["NC"]).reshape(-1, 1)
    df["NC"] = scaler.fit_transform(nc_X)

    ada2_X = np.array(df["ADA2"]).reshape(-1, 1)
    df["ADA2"] = scaler.fit_transform(ada2_X)

    adr1_X = np.array(df["ADR1"]).reshape(-1, 1)
    df["ADR1"] = scaler.fit_transform(adr1_X)

    aft2_X = np.array(df["AFT2"]).reshape(-1, 1)
    df["AFT2"] = scaler.fit_transform(aft2_X)

    PC1_X = np.array(df["PC1"]).reshape(-1, 1)
    df["PC1"] = scaler.fit_transform(PC1_X)

    PC2_X = np.array(df["PC2"]).reshape(-1, 1)
    df["PC2"] = scaler.fit_transform(PC2_X)

    PC3_X = np.array(df["PC3"]).reshape(-1, 1)
    df["PC3"] = scaler.fit_transform(PC3_X)

    SL_DC_SUM_X = np.array(df["SL_DC_SUM"]).reshape(-1, 1)
    df["SL_DC_SUM"] = scaler.fit_transform(SL_DC_SUM_X)

    SL_DC_SUB_X = np.array(df["SL_DC_SUB"]).reshape(-1, 1)
    df["SL_DC_SUB"] = scaler.fit_transform(SL_DC_SUB_X)

    SL_DC_PRO_X = np.array(df["SL_DC_PRO"]).reshape(-1, 1)
    df["SL_DC_PRO"] = scaler.fit_transform(SL_DC_PRO_X)

    SL_EC_SUM_X = np.array(df["SL_EC_SUM"]).reshape(-1, 1)
    df["SL_EC_SUM"] = scaler.fit_transform(SL_EC_SUM_X)

    SL_EC_SUB_X = np.array(df["SL_EC_SUB"]).reshape(-1, 1)
    df["SL_EC_SUB"] = scaler.fit_transform(SL_EC_SUB_X)

    SL_EC_PRO_X = np.array(df["SL_EC_PRO"]).reshape(-1, 1)
    df["SL_EC_PRO"] = scaler.fit_transform(SL_EC_PRO_X)

    SL_IC_SUM_X = np.array(df["SL_IC_SUM"]).reshape(-1, 1)
    df["SL_IC_SUM"] = scaler.fit_transform(SL_IC_SUM_X)

    SL_IC_SUB_X = np.array(df["SL_IC_SUB"]).reshape(-1, 1)
    df["SL_IC_SUB"] = scaler.fit_transform(SL_IC_SUB_X)

    SL_IC_PRO_X = np.array(df["SL_IC_PRO"]).reshape(-1, 1)
    df["SL_IC_PRO"] = scaler.fit_transform(SL_IC_PRO_X)

    SL_LAC_SUM_X = np.array(df["SL_LAC_SUM"]).reshape(-1, 1)
    df["SL_LAC_SUM"] = scaler.fit_transform(SL_LAC_SUM_X)

    SL_LAC_SUB_X = np.array(df["SL_LAC_SUB"]).reshape(-1, 1)
    df["SL_LAC_SUB"] = scaler.fit_transform(SL_LAC_SUB_X)

    SL_LAC_PRO_X = np.array(df["SL_LAC_PRO"]).reshape(-1, 1)
    df["SL_LAC_PRO"] = scaler.fit_transform(SL_LAC_PRO_X)

    SL_BC_SUM_X = np.array(df["SL_BC_SUM"]).reshape(-1, 1)
    df["SL_BC_SUM"] = scaler.fit_transform(SL_BC_SUM_X)

    SL_BC_SUB_X = np.array(df["SL_BC_SUB"]).reshape(-1, 1)
    df["SL_BC_SUB"] = scaler.fit_transform(SL_BC_SUB_X)

    SL_BC_PRO_X = np.array(df["SL_BC_PRO"]).reshape(-1, 1)
    df["SL_BC_PRO"] = scaler.fit_transform(SL_BC_PRO_X)

    SL_CC_SUM_X = np.array(df["SL_CC_SUM"]).reshape(-1, 1)
    df["SL_CC_SUM"] = scaler.fit_transform(SL_CC_SUM_X)

    SL_CC_SUB_X = np.array(df["SL_CC_SUB"]).reshape(-1, 1)
    df["SL_CC_SUB"] = scaler.fit_transform(SL_CC_SUB_X)

    SL_CC_PRO_X = np.array(df["SL_CC_PRO"]).reshape(-1, 1)
    df["SL_CC_PRO"] = scaler.fit_transform(SL_CC_PRO_X)

    SL_NC_SUM_X = np.array(df["SL_NC_SUM"]).reshape(-1, 1)
    df["SL_NC_SUM"] = scaler.fit_transform(SL_NC_SUM_X)

    SL_NC_SUB_X = np.array(df["SL_NC_SUB"]).reshape(-1, 1)
    df["SL_NC_SUB"] = scaler.fit_transform(SL_NC_SUB_X)

    SL_NC_PRO_X = np.array(df["SL_NC_PRO"]).reshape(-1, 1)
    df["SL_NC_PRO"] = scaler.fit_transform(SL_NC_PRO_X)

    SL_ADA2_SUM_X = np.array(df["SL_ADA2_SUM"]).reshape(-1, 1)
    df["SL_ADA2_SUM"] = scaler.fit_transform(SL_ADA2_SUM_X)

    SL_ADA2_SUB_X = np.array(df["SL_ADA2_SUB"]).reshape(-1, 1)
    df["SL_ADA2_SUB"] = scaler.fit_transform(SL_ADA2_SUB_X)

    SL_ADA2_PRO_X = np.array(df["SL_ADA2_PRO"]).reshape(-1, 1)
    df["SL_ADA2_PRO"] = scaler.fit_transform(SL_ADA2_PRO_X)

    SL_ADR1_SUM_X = np.array(df["SL_ADR1_SUM"]).reshape(-1, 1)
    df["SL_ADR1_SUM"] = scaler.fit_transform(SL_ADR1_SUM_X)

    SL_ADR1_SUB_X = np.array(df["SL_ADR1_SUB"]).reshape(-1, 1)
    df["SL_ADR1_SUB"] = scaler.fit_transform(SL_ADR1_SUB_X)

    SL_ADR1_PRO_X = np.array(df["SL_ADR1_PRO"]).reshape(-1, 1)
    df["SL_ADR1_PRO"] = scaler.fit_transform(SL_ADR1_PRO_X)

    SL_AFT2_SUM_X = np.array(df["SL_AFT2_SUM"]).reshape(-1, 1)
    df["SL_AFT2_SUM"] = scaler.fit_transform(SL_AFT2_SUM_X)

    SL_AFT2_SUB_X = np.array(df["SL_AFT2_SUB"]).reshape(-1, 1)
    df["SL_AFT2_SUB"] = scaler.fit_transform(SL_AFT2_SUB_X)

    SL_AFT2_PRO_X = np.array(df["SL_AFT2_PRO"]).reshape(-1, 1)
    df["SL_AFT2_PRO"] = scaler.fit_transform(SL_AFT2_PRO_X)

    SL_PC1_SUM_X = np.array(df["SL_PC1_SUM"]).reshape(-1, 1)
    df["SL_PC1_SUM"] = scaler.fit_transform(SL_PC1_SUM_X)

    SL_PC1_SUB_X = np.array(df["SL_PC1_SUB"]).reshape(-1, 1)
    df["SL_PC1_SUB"] = scaler.fit_transform(SL_PC1_SUB_X)

    SL_PC1_PRO_X = np.array(df["SL_PC1_PRO"]).reshape(-1, 1)
    df["SL_PC1_PRO"] = scaler.fit_transform(SL_PC1_PRO_X)

    SL_PC2_SUM_X = np.array(df["SL_PC2_SUM"]).reshape(-1, 1)
    df["SL_PC2_SUM"] = scaler.fit_transform(SL_PC2_SUM_X)

    SL_PC2_SUB_X = np.array(df["SL_PC2_SUB"]).reshape(-1, 1)
    df["SL_PC2_SUB"] = scaler.fit_transform(SL_PC2_SUB_X)

    SL_PC2_PRO_X = np.array(df["SL_PC2_PRO"]).reshape(-1, 1)
    df["SL_PC2_PRO"] = scaler.fit_transform(SL_PC2_PRO_X)

    SL_PC3_SUM_X = np.array(df["SL_PC3_SUM"]).reshape(-1, 1)
    df["SL_PC3_SUM"] = scaler.fit_transform(SL_PC3_SUM_X)

    SL_PC3_SUB_X = np.array(df["SL_PC3_SUB"]).reshape(-1, 1)
    df["SL_PC3_SUB"] = scaler.fit_transform(SL_PC3_SUB_X)

    SL_PC3_PRO_X = np.array(df["SL_PC3_PRO"]).reshape(-1, 1)
    df["SL_PC3_PRO"] = scaler.fit_transform(SL_PC3_PRO_X)

    DC_EC_SUM_X = np.array(df["DC_EC_SUM"]).reshape(-1, 1)
    df["DC_EC_SUM"] = scaler.fit_transform(DC_EC_SUM_X)

    DC_EC_SUB_X = np.array(df["DC_EC_SUB"]).reshape(-1, 1)
    df["DC_EC_SUB"] = scaler.fit_transform(DC_EC_SUB_X)

    DC_EC_PRO_X = np.array(df["DC_EC_PRO"]).reshape(-1, 1)
    df["DC_EC_PRO"] = scaler.fit_transform(DC_EC_PRO_X)

    DC_IC_SUM_X = np.array(df["DC_IC_SUM"]).reshape(-1, 1)
    df["DC_IC_SUM"] = scaler.fit_transform(DC_IC_SUM_X)

    DC_IC_SUB_X = np.array(df["DC_IC_SUB"]).reshape(-1, 1)
    df["DC_IC_SUB"] = scaler.fit_transform(DC_IC_SUB_X)

    DC_IC_PRO_X = np.array(df["DC_IC_PRO"]).reshape(-1, 1)
    df["DC_IC_PRO"] = scaler.fit_transform(DC_IC_PRO_X)

    DC_LAC_SUM_X = np.array(df["DC_LAC_SUM"]).reshape(-1, 1)
    df["DC_LAC_SUM"] = scaler.fit_transform(DC_LAC_SUM_X)

    DC_LAC_SUB_X = np.array(df["DC_LAC_SUB"]).reshape(-1, 1)
    df["DC_LAC_SUB"] = scaler.fit_transform(DC_LAC_SUB_X)

    DC_LAC_PRO_X = np.array(df["DC_LAC_PRO"]).reshape(-1, 1)
    df["DC_LAC_PRO"] = scaler.fit_transform(DC_LAC_PRO_X)

    DC_BC_SUM_X = np.array(df["DC_BC_SUM"]).reshape(-1, 1)
    df["DC_BC_SUM"] = scaler.fit_transform(DC_BC_SUM_X)

    DC_BC_SUB_X = np.array(df["DC_BC_SUB"]).reshape(-1, 1)
    df["DC_BC_SUB"] = scaler.fit_transform(DC_BC_SUB_X)

    DC_BC_PRO_X = np.array(df["DC_BC_PRO"]).reshape(-1, 1)
    df["DC_BC_PRO"] = scaler.fit_transform(DC_BC_PRO_X)

    DC_CC_SUM_X = np.array(df["DC_CC_SUM"]).reshape(-1, 1)
    df["DC_CC_SUM"] = scaler.fit_transform(DC_CC_SUM_X)

    DC_CC_SUB_X = np.array(df["DC_CC_SUB"]).reshape(-1, 1)
    df["DC_CC_SUB"] = scaler.fit_transform(DC_CC_SUB_X)

    DC_CC_PRO_X = np.array(df["DC_CC_PRO"]).reshape(-1, 1)
    df["DC_CC_PRO"] = scaler.fit_transform(DC_CC_PRO_X)

    DC_NC_SUM_X = np.array(df["DC_NC_SUM"]).reshape(-1, 1)
    df["DC_NC_SUM"] = scaler.fit_transform(DC_NC_SUM_X)

    DC_NC_SUB_X = np.array(df["DC_NC_SUB"]).reshape(-1, 1)
    df["DC_NC_SUB"] = scaler.fit_transform(DC_NC_SUB_X)

    DC_NC_PRO_X = np.array(df["DC_NC_PRO"]).reshape(-1, 1)
    df["DC_NC_PRO"] = scaler.fit_transform(DC_NC_PRO_X)

    DC_ADA2_SUM_X = np.array(df["DC_ADA2_SUM"]).reshape(-1, 1)
    df["DC_ADA2_SUM"] = scaler.fit_transform(DC_ADA2_SUM_X)

    DC_ADA2_SUB_X = np.array(df["DC_ADA2_SUB"]).reshape(-1, 1)
    df["DC_ADA2_SUB"] = scaler.fit_transform(DC_ADA2_SUB_X)

    DC_ADA2_PRO_X = np.array(df["DC_ADA2_PRO"]).reshape(-1, 1)
    df["DC_ADA2_PRO"] = scaler.fit_transform(DC_ADA2_PRO_X)

    DC_ADR1_SUM_X = np.array(df["DC_ADR1_SUM"]).reshape(-1, 1)
    df["DC_ADR1_SUM"] = scaler.fit_transform(DC_ADR1_SUM_X)

    DC_ADR1_SUB_X = np.array(df["DC_ADR1_SUB"]).reshape(-1, 1)
    df["DC_ADR1_SUB"] = scaler.fit_transform(DC_ADR1_SUB_X)

    DC_ADR1_PRO_X = np.array(df["DC_ADR1_PRO"]).reshape(-1, 1)
    df["DC_ADR1_PRO"] = scaler.fit_transform(DC_ADR1_PRO_X)

    DC_AFT2_SUM_X = np.array(df["DC_AFT2_SUM"]).reshape(-1, 1)
    df["DC_AFT2_SUM"] = scaler.fit_transform(DC_AFT2_SUM_X)

    DC_AFT2_SUB_X = np.array(df["DC_AFT2_SUB"]).reshape(-1, 1)
    df["DC_AFT2_SUB"] = scaler.fit_transform(DC_AFT2_SUB_X)

    DC_AFT2_PRO_X = np.array(df["DC_AFT2_PRO"]).reshape(-1, 1)
    df["DC_AFT2_PRO"] = scaler.fit_transform(DC_AFT2_PRO_X)

    DC_PC1_SUM_X = np.array(df["DC_PC1_SUM"]).reshape(-1, 1)
    df["DC_PC1_SUM"] = scaler.fit_transform(DC_PC1_SUM_X)

    DC_PC1_SUB_X = np.array(df["DC_PC1_SUB"]).reshape(-1, 1)
    df["DC_PC1_SUB"] = scaler.fit_transform(DC_PC1_SUB_X)

    DC_PC1_PRO_X = np.array(df["DC_PC1_PRO"]).reshape(-1, 1)
    df["DC_PC1_PRO"] = scaler.fit_transform(DC_PC1_PRO_X)

    DC_PC2_SUM_X = np.array(df["DC_PC2_SUM"]).reshape(-1, 1)
    df["DC_PC2_SUM"] = scaler.fit_transform(DC_PC2_SUM_X)

    DC_PC2_SUB_X = np.array(df["DC_PC2_SUB"]).reshape(-1, 1)
    df["DC_PC2_SUB"] = scaler.fit_transform(DC_PC2_SUB_X)

    DC_PC2_PRO_X = np.array(df["DC_PC2_PRO"]).reshape(-1, 1)
    df["DC_PC2_PRO"] = scaler.fit_transform(DC_PC2_PRO_X)

    DC_PC3_SUM_X = np.array(df["DC_PC3_SUM"]).reshape(-1, 1)
    df["DC_PC3_SUM"] = scaler.fit_transform(DC_PC3_SUM_X)

    DC_PC3_SUB_X = np.array(df["DC_PC3_SUB"]).reshape(-1, 1)
    df["DC_PC3_SUB"] = scaler.fit_transform(DC_PC3_SUB_X)

    DC_PC3_PRO_X = np.array(df["DC_PC3_PRO"]).reshape(-1, 1)
    df["DC_PC3_PRO"] = scaler.fit_transform(DC_PC3_PRO_X)

    EC_IC_SUM_X = np.array(df["EC_IC_SUM"]).reshape(-1, 1)
    df["EC_IC_SUM"] = scaler.fit_transform(EC_IC_SUM_X)

    EC_IC_SUB_X = np.array(df["EC_IC_SUB"]).reshape(-1, 1)
    df["EC_IC_SUB"] = scaler.fit_transform(EC_IC_SUB_X)

    EC_IC_PRO_X = np.array(df["EC_IC_PRO"]).reshape(-1, 1)
    df["EC_IC_PRO"] = scaler.fit_transform(EC_IC_PRO_X)

    EC_LAC_SUM_X = np.array(df["EC_LAC_SUM"]).reshape(-1, 1)
    df["EC_LAC_SUM"] = scaler.fit_transform(EC_LAC_SUM_X)

    EC_LAC_SUB_X = np.array(df["EC_LAC_SUB"]).reshape(-1, 1)
    df["EC_LAC_SUB"] = scaler.fit_transform(EC_LAC_SUB_X)

    EC_LAC_PRO_X = np.array(df["EC_LAC_PRO"]).reshape(-1, 1)
    df["EC_LAC_PRO"] = scaler.fit_transform(EC_LAC_PRO_X)

    EC_BC_SUM_X = np.array(df["EC_BC_SUM"]).reshape(-1, 1)
    df["EC_BC_SUM"] = scaler.fit_transform(EC_BC_SUM_X)

    EC_BC_SUB_X = np.array(df["EC_BC_SUB"]).reshape(-1, 1)
    df["EC_BC_SUB"] = scaler.fit_transform(EC_BC_SUB_X)

    EC_BC_PRO_X = np.array(df["EC_BC_PRO"]).reshape(-1, 1)
    df["EC_BC_PRO"] = scaler.fit_transform(EC_BC_PRO_X)

    EC_CC_SUM_X = np.array(df["EC_CC_SUM"]).reshape(-1, 1)
    df["EC_CC_SUM"] = scaler.fit_transform(EC_CC_SUM_X)

    EC_CC_SUB_X = np.array(df["EC_CC_SUB"]).reshape(-1, 1)
    df["EC_CC_SUB"] = scaler.fit_transform(EC_CC_SUB_X)

    EC_CC_PRO_X = np.array(df["EC_CC_PRO"]).reshape(-1, 1)
    df["EC_CC_PRO"] = scaler.fit_transform(EC_CC_PRO_X)

    EC_NC_SUM_X = np.array(df["EC_NC_SUM"]).reshape(-1, 1)
    df["EC_NC_SUM"] = scaler.fit_transform(EC_NC_SUM_X)

    EC_NC_SUB_X = np.array(df["EC_NC_SUB"]).reshape(-1, 1)
    df["EC_NC_SUB"] = scaler.fit_transform(EC_NC_SUB_X)

    EC_NC_PRO_X = np.array(df["EC_NC_PRO"]).reshape(-1, 1)
    df["EC_NC_PRO"] = scaler.fit_transform(EC_NC_PRO_X)

    EC_ADA2_SUM_X = np.array(df["EC_ADA2_SUM"]).reshape(-1, 1)
    df["EC_ADA2_SUM"] = scaler.fit_transform(EC_ADA2_SUM_X)

    EC_ADA2_SUB_X = np.array(df["EC_ADA2_SUB"]).reshape(-1, 1)
    df["EC_ADA2_SUB"] = scaler.fit_transform(EC_ADA2_SUB_X)

    EC_ADA2_PRO_X = np.array(df["EC_ADA2_PRO"]).reshape(-1, 1)
    df["EC_ADA2_PRO"] = scaler.fit_transform(EC_ADA2_PRO_X)

    EC_ADR1_SUM_X = np.array(df["EC_ADR1_SUM"]).reshape(-1, 1)
    df["EC_ADR1_SUM"] = scaler.fit_transform(EC_ADR1_SUM_X)

    EC_ADR1_SUB_X = np.array(df["EC_ADR1_SUB"]).reshape(-1, 1)
    df["EC_ADR1_SUB"] = scaler.fit_transform(EC_ADR1_SUB_X)

    EC_ADR1_PRO_X = np.array(df["EC_ADR1_PRO"]).reshape(-1, 1)
    df["EC_ADR1_PRO"] = scaler.fit_transform(EC_ADR1_PRO_X)

    EC_AFT2_SUM_X = np.array(df["EC_AFT2_SUM"]).reshape(-1, 1)
    df["EC_AFT2_SUM"] = scaler.fit_transform(EC_AFT2_SUM_X)

    EC_AFT2_SUB_X = np.array(df["EC_AFT2_SUB"]).reshape(-1, 1)
    df["EC_AFT2_SUB"] = scaler.fit_transform(EC_AFT2_SUB_X)

    EC_AFT2_PRO_X = np.array(df["EC_AFT2_PRO"]).reshape(-1, 1)
    df["EC_AFT2_PRO"] = scaler.fit_transform(EC_AFT2_PRO_X)

    EC_PC1_SUM_X = np.array(df["EC_PC1_SUM"]).reshape(-1, 1)
    df["EC_PC1_SUM"] = scaler.fit_transform(EC_PC1_SUM_X)

    EC_PC1_SUB_X = np.array(df["EC_PC1_SUB"]).reshape(-1, 1)
    df["EC_PC1_SUB"] = scaler.fit_transform(EC_PC1_SUB_X)

    EC_PC1_PRO_X = np.array(df["EC_PC1_PRO"]).reshape(-1, 1)
    df["EC_PC1_PRO"] = scaler.fit_transform(EC_PC1_PRO_X)

    EC_PC2_SUM_X = np.array(df["EC_PC2_SUM"]).reshape(-1, 1)
    df["EC_PC2_SUM"] = scaler.fit_transform(EC_PC2_SUM_X)

    EC_PC2_SUB_X = np.array(df["EC_PC2_SUB"]).reshape(-1, 1)
    df["EC_PC2_SUB"] = scaler.fit_transform(EC_PC2_SUB_X)

    EC_PC2_PRO_X = np.array(df["EC_PC2_PRO"]).reshape(-1, 1)
    df["EC_PC2_PRO"] = scaler.fit_transform(EC_PC2_PRO_X)

    EC_PC3_SUM_X = np.array(df["EC_PC3_SUM"]).reshape(-1, 1)
    df["EC_PC3_SUM"] = scaler.fit_transform(EC_PC3_SUM_X)

    EC_PC3_SUB_X = np.array(df["EC_PC3_SUB"]).reshape(-1, 1)
    df["EC_PC3_SUB"] = scaler.fit_transform(EC_PC3_SUB_X)

    EC_PC3_PRO_X = np.array(df["EC_PC3_PRO"]).reshape(-1, 1)
    df["EC_PC3_PRO"] = scaler.fit_transform(EC_PC3_PRO_X)

    IC_LAC_SUM_X = np.array(df["IC_LAC_SUM"]).reshape(-1, 1)
    df["IC_LAC_SUM"] = scaler.fit_transform(IC_LAC_SUM_X)

    IC_LAC_SUB_X = np.array(df["IC_LAC_SUB"]).reshape(-1, 1)
    df["IC_LAC_SUB"] = scaler.fit_transform(IC_LAC_SUB_X)

    IC_LAC_PRO_X = np.array(df["IC_LAC_PRO"]).reshape(-1, 1)
    df["IC_LAC_PRO"] = scaler.fit_transform(IC_LAC_PRO_X)

    IC_BC_SUM_X = np.array(df["IC_BC_SUM"]).reshape(-1, 1)
    df["IC_BC_SUM"] = scaler.fit_transform(IC_BC_SUM_X)

    IC_BC_SUB_X = np.array(df["IC_BC_SUB"]).reshape(-1, 1)
    df["IC_BC_SUB"] = scaler.fit_transform(IC_BC_SUB_X)

    IC_BC_PRO_X = np.array(df["IC_BC_PRO"]).reshape(-1, 1)
    df["IC_BC_PRO"] = scaler.fit_transform(IC_BC_PRO_X)

    IC_CC_SUM_X = np.array(df["IC_CC_SUM"]).reshape(-1, 1)
    df["IC_CC_SUM"] = scaler.fit_transform(IC_CC_SUM_X)

    IC_CC_SUB_X = np.array(df["IC_CC_SUB"]).reshape(-1, 1)
    df["IC_CC_SUB"] = scaler.fit_transform(IC_CC_SUB_X)

    IC_CC_PRO_X = np.array(df["IC_CC_PRO"]).reshape(-1, 1)
    df["IC_CC_PRO"] = scaler.fit_transform(IC_CC_PRO_X)

    IC_NC_SUM_X = np.array(df["IC_NC_SUM"]).reshape(-1, 1)
    df["IC_NC_SUM"] = scaler.fit_transform(IC_NC_SUM_X)

    IC_NC_SUB_X = np.array(df["IC_NC_SUB"]).reshape(-1, 1)
    df["IC_NC_SUB"] = scaler.fit_transform(IC_NC_SUB_X)

    IC_NC_PRO_X = np.array(df["IC_NC_PRO"]).reshape(-1, 1)
    df["IC_NC_PRO"] = scaler.fit_transform(IC_NC_PRO_X)

    IC_ADA2_SUM_X = np.array(df["IC_ADA2_SUM"]).reshape(-1, 1)
    df["IC_ADA2_SUM"] = scaler.fit_transform(IC_ADA2_SUM_X)

    IC_ADA2_SUB_X = np.array(df["IC_ADA2_SUB"]).reshape(-1, 1)
    df["IC_ADA2_SUB"] = scaler.fit_transform(IC_ADA2_SUB_X)

    IC_ADA2_PRO_X = np.array(df["IC_ADA2_PRO"]).reshape(-1, 1)
    df["IC_ADA2_PRO"] = scaler.fit_transform(IC_ADA2_PRO_X)

    IC_ADR1_SUM_X = np.array(df["IC_ADR1_SUM"]).reshape(-1, 1)
    df["IC_ADR1_SUM"] = scaler.fit_transform(IC_ADR1_SUM_X)

    IC_ADR1_SUB_X = np.array(df["IC_ADR1_SUB"]).reshape(-1, 1)
    df["IC_ADR1_SUB"] = scaler.fit_transform(IC_ADR1_SUB_X)

    IC_ADR1_PRO_X = np.array(df["IC_ADR1_PRO"]).reshape(-1, 1)
    df["IC_ADR1_PRO"] = scaler.fit_transform(IC_ADR1_PRO_X)

    IC_AFT2_SUM_X = np.array(df["IC_AFT2_SUM"]).reshape(-1, 1)
    df["IC_AFT2_SUM"] = scaler.fit_transform(IC_AFT2_SUM_X)

    IC_AFT2_SUB_X = np.array(df["IC_AFT2_SUB"]).reshape(-1, 1)
    df["IC_AFT2_SUB"] = scaler.fit_transform(IC_AFT2_SUB_X)

    IC_AFT2_PRO_X = np.array(df["IC_AFT2_PRO"]).reshape(-1, 1)
    df["IC_AFT2_PRO"] = scaler.fit_transform(IC_AFT2_PRO_X)

    IC_PC1_SUM_X = np.array(df["IC_PC1_SUM"]).reshape(-1, 1)
    df["IC_PC1_SUM"] = scaler.fit_transform(IC_PC1_SUM_X)

    IC_PC1_SUB_X = np.array(df["IC_PC1_SUB"]).reshape(-1, 1)
    df["IC_PC1_SUB"] = scaler.fit_transform(IC_PC1_SUB_X)

    IC_PC1_PRO_X = np.array(df["IC_PC1_PRO"]).reshape(-1, 1)
    df["IC_PC1_PRO"] = scaler.fit_transform(IC_PC1_PRO_X)

    IC_PC2_SUM_X = np.array(df["IC_PC2_SUM"]).reshape(-1, 1)
    df["IC_PC2_SUM"] = scaler.fit_transform(IC_PC2_SUM_X)

    IC_PC2_SUB_X = np.array(df["IC_PC2_SUB"]).reshape(-1, 1)
    df["IC_PC2_SUB"] = scaler.fit_transform(IC_PC2_SUB_X)

    IC_PC2_PRO_X = np.array(df["IC_PC2_PRO"]).reshape(-1, 1)
    df["IC_PC2_PRO"] = scaler.fit_transform(IC_PC2_PRO_X)

    IC_PC3_SUM_X = np.array(df["IC_PC3_SUM"]).reshape(-1, 1)
    df["IC_PC3_SUM"] = scaler.fit_transform(IC_PC3_SUM_X)

    IC_PC3_SUB_X = np.array(df["IC_PC3_SUB"]).reshape(-1, 1)
    df["IC_PC3_SUB"] = scaler.fit_transform(IC_PC3_SUB_X)

    IC_PC3_PRO_X = np.array(df["IC_PC3_PRO"]).reshape(-1, 1)
    df["IC_PC3_PRO"] = scaler.fit_transform(IC_PC3_PRO_X)

    LAC_BC_SUM_X = np.array(df["LAC_BC_SUM"]).reshape(-1, 1)
    df["LAC_BC_SUM"] = scaler.fit_transform(LAC_BC_SUM_X)

    LAC_BC_SUB_X = np.array(df["LAC_BC_SUB"]).reshape(-1, 1)
    df["LAC_BC_SUB"] = scaler.fit_transform(LAC_BC_SUB_X)

    LAC_BC_PRO_X = np.array(df["LAC_BC_PRO"]).reshape(-1, 1)
    df["LAC_BC_PRO"] = scaler.fit_transform(LAC_BC_PRO_X)

    LAC_CC_SUM_X = np.array(df["LAC_CC_SUM"]).reshape(-1, 1)
    df["LAC_CC_SUM"] = scaler.fit_transform(LAC_CC_SUM_X)

    LAC_CC_SUB_X = np.array(df["LAC_CC_SUB"]).reshape(-1, 1)
    df["LAC_CC_SUB"] = scaler.fit_transform(LAC_CC_SUB_X)

    LAC_CC_PRO_X = np.array(df["LAC_CC_PRO"]).reshape(-1, 1)
    df["LAC_CC_PRO"] = scaler.fit_transform(LAC_CC_PRO_X)

    LAC_NC_SUM_X = np.array(df["LAC_NC_SUM"]).reshape(-1, 1)
    df["LAC_NC_SUM"] = scaler.fit_transform(LAC_NC_SUM_X)

    LAC_NC_SUB_X = np.array(df["LAC_NC_SUB"]).reshape(-1, 1)
    df["LAC_NC_SUB"] = scaler.fit_transform(LAC_NC_SUB_X)

    LAC_NC_PRO_X = np.array(df["LAC_NC_PRO"]).reshape(-1, 1)
    df["LAC_NC_PRO"] = scaler.fit_transform(LAC_NC_PRO_X)

    LAC_ADA2_SUM_X = np.array(df["LAC_ADA2_SUM"]).reshape(-1, 1)
    df["LAC_ADA2_SUM"] = scaler.fit_transform(LAC_ADA2_SUM_X)

    LAC_ADA2_SUB_X = np.array(df["LAC_ADA2_SUB"]).reshape(-1, 1)
    df["LAC_ADA2_SUB"] = scaler.fit_transform(LAC_ADA2_SUB_X)

    LAC_ADA2_PRO_X = np.array(df["LAC_ADA2_PRO"]).reshape(-1, 1)
    df["LAC_ADA2_PRO"] = scaler.fit_transform(LAC_ADA2_PRO_X)

    LAC_ADR1_SUM_X = np.array(df["LAC_ADR1_SUM"]).reshape(-1, 1)
    df["LAC_ADR1_SUM"] = scaler.fit_transform(LAC_ADR1_SUM_X)

    LAC_ADR1_SUB_X = np.array(df["LAC_ADR1_SUB"]).reshape(-1, 1)
    df["LAC_ADR1_SUB"] = scaler.fit_transform(LAC_ADR1_SUB_X)

    LAC_ADR1_PRO_X = np.array(df["LAC_ADR1_PRO"]).reshape(-1, 1)
    df["LAC_ADR1_PRO"] = scaler.fit_transform(LAC_ADR1_PRO_X)

    LAC_AFT2_SUM_X = np.array(df["LAC_AFT2_SUM"]).reshape(-1, 1)
    df["LAC_AFT2_SUM"] = scaler.fit_transform(LAC_AFT2_SUM_X)

    LAC_AFT2_SUB_X = np.array(df["LAC_AFT2_SUB"]).reshape(-1, 1)
    df["LAC_AFT2_SUB"] = scaler.fit_transform(LAC_AFT2_SUB_X)

    LAC_AFT2_PRO_X = np.array(df["LAC_AFT2_PRO"]).reshape(-1, 1)
    df["LAC_AFT2_PRO"] = scaler.fit_transform(LAC_AFT2_PRO_X)

    LAC_PC1_SUM_X = np.array(df["LAC_PC1_SUM"]).reshape(-1, 1)
    df["LAC_PC1_SUM"] = scaler.fit_transform(LAC_PC1_SUM_X)

    LAC_PC1_SUB_X = np.array(df["LAC_PC1_SUB"]).reshape(-1, 1)
    df["LAC_PC1_SUB"] = scaler.fit_transform(LAC_PC1_SUB_X)

    LAC_PC1_PRO_X = np.array(df["LAC_PC1_PRO"]).reshape(-1, 1)
    df["LAC_PC1_PRO"] = scaler.fit_transform(LAC_PC1_PRO_X)

    LAC_PC2_SUM_X = np.array(df["LAC_PC2_SUM"]).reshape(-1, 1)
    df["LAC_PC2_SUM"] = scaler.fit_transform(LAC_PC2_SUM_X)

    LAC_PC2_SUB_X = np.array(df["LAC_PC2_SUB"]).reshape(-1, 1)
    df["LAC_PC2_SUB"] = scaler.fit_transform(LAC_PC2_SUB_X)

    LAC_PC2_PRO_X = np.array(df["LAC_PC2_PRO"]).reshape(-1, 1)
    df["LAC_PC2_PRO"] = scaler.fit_transform(LAC_PC2_PRO_X)

    LAC_PC3_SUM_X = np.array(df["LAC_PC3_SUM"]).reshape(-1, 1)
    df["LAC_PC3_SUM"] = scaler.fit_transform(LAC_PC3_SUM_X)

    LAC_PC3_SUB_X = np.array(df["LAC_PC3_SUB"]).reshape(-1, 1)
    df["LAC_PC3_SUB"] = scaler.fit_transform(LAC_PC3_SUB_X)

    LAC_PC3_PRO_X = np.array(df["LAC_PC3_PRO"]).reshape(-1, 1)
    df["LAC_PC3_PRO"] = scaler.fit_transform(LAC_PC3_PRO_X)

    BC_CC_SUM_X = np.array(df["BC_CC_SUM"]).reshape(-1, 1)
    df["BC_CC_SUM"] = scaler.fit_transform(BC_CC_SUM_X)

    BC_CC_SUB_X = np.array(df["BC_CC_SUB"]).reshape(-1, 1)
    df["BC_CC_SUB"] = scaler.fit_transform(BC_CC_SUB_X)

    BC_CC_PRO_X = np.array(df["BC_CC_PRO"]).reshape(-1, 1)
    df["BC_CC_PRO"] = scaler.fit_transform(BC_CC_PRO_X)

    BC_NC_SUM_X = np.array(df["BC_NC_SUM"]).reshape(-1, 1)
    df["BC_NC_SUM"] = scaler.fit_transform(BC_NC_SUM_X)

    BC_NC_SUB_X = np.array(df["BC_NC_SUB"]).reshape(-1, 1)
    df["BC_NC_SUB"] = scaler.fit_transform(BC_NC_SUB_X)

    BC_NC_PRO_X = np.array(df["BC_NC_PRO"]).reshape(-1, 1)
    df["BC_NC_PRO"] = scaler.fit_transform(BC_NC_PRO_X)

    BC_ADA2_SUM_X = np.array(df["BC_ADA2_SUM"]).reshape(-1, 1)
    df["BC_ADA2_SUM"] = scaler.fit_transform(BC_ADA2_SUM_X)

    BC_ADA2_SUB_X = np.array(df["BC_ADA2_SUB"]).reshape(-1, 1)
    df["BC_ADA2_SUB"] = scaler.fit_transform(BC_ADA2_SUB_X)

    BC_ADA2_PRO_X = np.array(df["BC_ADA2_PRO"]).reshape(-1, 1)
    df["BC_ADA2_PRO"] = scaler.fit_transform(BC_ADA2_PRO_X)

    BC_ADR1_SUM_X = np.array(df["BC_ADR1_SUM"]).reshape(-1, 1)
    df["BC_ADR1_SUM"] = scaler.fit_transform(BC_ADR1_SUM_X)

    BC_ADR1_SUB_X = np.array(df["BC_ADR1_SUB"]).reshape(-1, 1)
    df["BC_ADR1_SUB"] = scaler.fit_transform(BC_ADR1_SUB_X)

    BC_ADR1_PRO_X = np.array(df["BC_ADR1_PRO"]).reshape(-1, 1)
    df["BC_ADR1_PRO"] = scaler.fit_transform(BC_ADR1_PRO_X)

    BC_AFT2_SUM_X = np.array(df["BC_AFT2_SUM"]).reshape(-1, 1)
    df["BC_AFT2_SUM"] = scaler.fit_transform(BC_AFT2_SUM_X)

    BC_AFT2_SUB_X = np.array(df["BC_AFT2_SUB"]).reshape(-1, 1)
    df["BC_AFT2_SUB"] = scaler.fit_transform(BC_AFT2_SUB_X)

    BC_AFT2_PRO_X = np.array(df["BC_AFT2_PRO"]).reshape(-1, 1)
    df["BC_AFT2_PRO"] = scaler.fit_transform(BC_AFT2_PRO_X)

    BC_PC1_SUM_X = np.array(df["BC_PC1_SUM"]).reshape(-1, 1)
    df["BC_PC1_SUM"] = scaler.fit_transform(BC_PC1_SUM_X)

    BC_PC1_SUB_X = np.array(df["BC_PC1_SUB"]).reshape(-1, 1)
    df["BC_PC1_SUB"] = scaler.fit_transform(BC_PC1_SUB_X)

    BC_PC1_PRO_X = np.array(df["BC_PC1_PRO"]).reshape(-1, 1)
    df["BC_PC1_PRO"] = scaler.fit_transform(BC_PC1_PRO_X)

    BC_PC2_SUM_X = np.array(df["BC_PC2_SUM"]).reshape(-1, 1)
    df["BC_PC2_SUM"] = scaler.fit_transform(BC_PC2_SUM_X)

    BC_PC2_SUB_X = np.array(df["BC_PC2_SUB"]).reshape(-1, 1)
    df["BC_PC2_SUB"] = scaler.fit_transform(BC_PC2_SUB_X)

    BC_PC2_PRO_X = np.array(df["BC_PC2_PRO"]).reshape(-1, 1)
    df["BC_PC2_PRO"] = scaler.fit_transform(BC_PC2_PRO_X)

    BC_PC3_SUM_X = np.array(df["BC_PC3_SUM"]).reshape(-1, 1)
    df["BC_PC3_SUM"] = scaler.fit_transform(BC_PC3_SUM_X)

    BC_PC3_SUB_X = np.array(df["BC_PC3_SUB"]).reshape(-1, 1)
    df["BC_PC3_SUB"] = scaler.fit_transform(BC_PC3_SUB_X)

    BC_PC3_PRO_X = np.array(df["BC_PC3_PRO"]).reshape(-1, 1)
    df["BC_PC3_PRO"] = scaler.fit_transform(BC_PC3_PRO_X)

    CC_NC_SUM_X = np.array(df["CC_NC_SUM"]).reshape(-1, 1)
    df["CC_NC_SUM"] = scaler.fit_transform(CC_NC_SUM_X)

    CC_NC_SUB_X = np.array(df["CC_NC_SUB"]).reshape(-1, 1)
    df["CC_NC_SUB"] = scaler.fit_transform(CC_NC_SUB_X)

    CC_NC_PRO_X = np.array(df["CC_NC_PRO"]).reshape(-1, 1)
    df["CC_NC_PRO"] = scaler.fit_transform(CC_NC_PRO_X)

    CC_ADA2_SUM_X = np.array(df["CC_ADA2_SUM"]).reshape(-1, 1)
    df["CC_ADA2_SUM"] = scaler.fit_transform(CC_ADA2_SUM_X)

    CC_ADA2_SUB_X = np.array(df["CC_ADA2_SUB"]).reshape(-1, 1)
    df["CC_ADA2_SUB"] = scaler.fit_transform(CC_ADA2_SUB_X)

    CC_ADA2_PRO_X = np.array(df["CC_ADA2_PRO"]).reshape(-1, 1)
    df["CC_ADA2_PRO"] = scaler.fit_transform(CC_ADA2_PRO_X)

    CC_ADR1_SUM_X = np.array(df["CC_ADR1_SUM"]).reshape(-1, 1)
    df["CC_ADR1_SUM"] = scaler.fit_transform(CC_ADR1_SUM_X)

    CC_ADR1_SUB_X = np.array(df["CC_ADR1_SUB"]).reshape(-1, 1)
    df["CC_ADR1_SUB"] = scaler.fit_transform(CC_ADR1_SUB_X)

    CC_ADR1_PRO_X = np.array(df["CC_ADR1_PRO"]).reshape(-1, 1)
    df["CC_ADR1_PRO"] = scaler.fit_transform(CC_ADR1_PRO_X)

    CC_AFT2_SUM_X = np.array(df["CC_AFT2_SUM"]).reshape(-1, 1)
    df["CC_AFT2_SUM"] = scaler.fit_transform(CC_AFT2_SUM_X)

    CC_AFT2_SUB_X = np.array(df["CC_AFT2_SUB"]).reshape(-1, 1)
    df["CC_AFT2_SUB"] = scaler.fit_transform(CC_AFT2_SUB_X)

    CC_AFT2_PRO_X = np.array(df["CC_AFT2_PRO"]).reshape(-1, 1)
    df["CC_AFT2_PRO"] = scaler.fit_transform(CC_AFT2_PRO_X)

    CC_PC1_SUM_X = np.array(df["CC_PC1_SUM"]).reshape(-1, 1)
    df["CC_PC1_SUM"] = scaler.fit_transform(CC_PC1_SUM_X)

    CC_PC1_SUB_X = np.array(df["CC_PC1_SUB"]).reshape(-1, 1)
    df["CC_PC1_SUB"] = scaler.fit_transform(CC_PC1_SUB_X)

    CC_PC1_PRO_X = np.array(df["CC_PC1_PRO"]).reshape(-1, 1)
    df["CC_PC1_PRO"] = scaler.fit_transform(CC_PC1_PRO_X)

    CC_PC2_SUM_X = np.array(df["CC_PC2_SUM"]).reshape(-1, 1)
    df["CC_PC2_SUM"] = scaler.fit_transform(CC_PC2_SUM_X)

    CC_PC2_SUB_X = np.array(df["CC_PC2_SUB"]).reshape(-1, 1)
    df["CC_PC2_SUB"] = scaler.fit_transform(CC_PC2_SUB_X)

    CC_PC2_PRO_X = np.array(df["CC_PC2_PRO"]).reshape(-1, 1)
    df["CC_PC2_PRO"] = scaler.fit_transform(CC_PC2_PRO_X)

    CC_PC3_SUM_X = np.array(df["CC_PC3_SUM"]).reshape(-1, 1)
    df["CC_PC3_SUM"] = scaler.fit_transform(CC_PC3_SUM_X)

    CC_PC3_SUB_X = np.array(df["CC_PC3_SUB"]).reshape(-1, 1)
    df["CC_PC3_SUB"] = scaler.fit_transform(CC_PC3_SUB_X)

    CC_PC3_PRO_X = np.array(df["CC_PC3_PRO"]).reshape(-1, 1)
    df["CC_PC3_PRO"] = scaler.fit_transform(CC_PC3_PRO_X)

    NC_ADA2_SUM_X = np.array(df["NC_ADA2_SUM"]).reshape(-1, 1)
    df["NC_ADA2_SUM"] = scaler.fit_transform(NC_ADA2_SUM_X)

    NC_ADA2_SUB_X = np.array(df["NC_ADA2_SUB"]).reshape(-1, 1)
    df["NC_ADA2_SUB"] = scaler.fit_transform(NC_ADA2_SUB_X)

    NC_ADA2_PRO_X = np.array(df["NC_ADA2_PRO"]).reshape(-1, 1)
    df["NC_ADA2_PRO"] = scaler.fit_transform(NC_ADA2_PRO_X)

    NC_ADR1_SUM_X = np.array(df["NC_ADR1_SUM"]).reshape(-1, 1)
    df["NC_ADR1_SUM"] = scaler.fit_transform(NC_ADR1_SUM_X)

    NC_ADR1_SUB_X = np.array(df["NC_ADR1_SUB"]).reshape(-1, 1)
    df["NC_ADR1_SUB"] = scaler.fit_transform(NC_ADR1_SUB_X)

    NC_ADR1_PRO_X = np.array(df["NC_ADR1_PRO"]).reshape(-1, 1)
    df["NC_ADR1_PRO"] = scaler.fit_transform(NC_ADR1_PRO_X)

    NC_AFT2_SUM_X = np.array(df["NC_AFT2_SUM"]).reshape(-1, 1)
    df["NC_AFT2_SUM"] = scaler.fit_transform(NC_AFT2_SUM_X)

    NC_AFT2_SUB_X = np.array(df["NC_AFT2_SUB"]).reshape(-1, 1)
    df["NC_AFT2_SUB"] = scaler.fit_transform(NC_AFT2_SUB_X)

    NC_AFT2_PRO_X = np.array(df["NC_AFT2_PRO"]).reshape(-1, 1)
    df["NC_AFT2_PRO"] = scaler.fit_transform(NC_AFT2_PRO_X)

    NC_PC1_SUM_X = np.array(df["NC_PC1_SUM"]).reshape(-1, 1)
    df["NC_PC1_SUM"] = scaler.fit_transform(NC_PC1_SUM_X)

    NC_PC1_SUB_X = np.array(df["NC_PC1_SUB"]).reshape(-1, 1)
    df["NC_PC1_SUB"] = scaler.fit_transform(NC_PC1_SUB_X)

    NC_PC1_PRO_X = np.array(df["NC_PC1_PRO"]).reshape(-1, 1)
    df["NC_PC1_PRO"] = scaler.fit_transform(NC_PC1_PRO_X)

    NC_PC2_SUM_X = np.array(df["NC_PC2_SUM"]).reshape(-1, 1)
    df["NC_PC2_SUM"] = scaler.fit_transform(NC_PC2_SUM_X)

    NC_PC2_SUB_X = np.array(df["NC_PC2_SUB"]).reshape(-1, 1)
    df["NC_PC2_SUB"] = scaler.fit_transform(NC_PC2_SUB_X)

    NC_PC2_PRO_X = np.array(df["NC_PC2_PRO"]).reshape(-1, 1)
    df["NC_PC2_PRO"] = scaler.fit_transform(NC_PC2_PRO_X)

    NC_PC3_SUM_X = np.array(df["NC_PC3_SUM"]).reshape(-1, 1)
    df["NC_PC3_SUM"] = scaler.fit_transform(NC_PC3_SUM_X)

    NC_PC3_SUB_X = np.array(df["NC_PC3_SUB"]).reshape(-1, 1)
    df["NC_PC3_SUB"] = scaler.fit_transform(NC_PC3_SUB_X)

    NC_PC3_PRO_X = np.array(df["NC_PC3_PRO"]).reshape(-1, 1)
    df["NC_PC3_PRO"] = scaler.fit_transform(NC_PC3_PRO_X)

    ADA2_ADR1_SUM_X = np.array(df["ADA2_ADR1_SUM"]).reshape(-1, 1)
    df["ADA2_ADR1_SUM"] = scaler.fit_transform(ADA2_ADR1_SUM_X)

    ADA2_ADR1_SUB_X = np.array(df["ADA2_ADR1_SUB"]).reshape(-1, 1)
    df["ADA2_ADR1_SUB"] = scaler.fit_transform(ADA2_ADR1_SUB_X)

    ADA2_ADR1_PRO_X = np.array(df["ADA2_ADR1_PRO"]).reshape(-1, 1)
    df["ADA2_ADR1_PRO"] = scaler.fit_transform(ADA2_ADR1_PRO_X)

    ADA2_AFT2_SUM_X = np.array(df["ADA2_AFT2_SUM"]).reshape(-1, 1)
    df["ADA2_AFT2_SUM"] = scaler.fit_transform(ADA2_AFT2_SUM_X)

    ADA2_AFT2_SUB_X = np.array(df["ADA2_AFT2_SUB"]).reshape(-1, 1)
    df["ADA2_AFT2_SUB"] = scaler.fit_transform(ADA2_AFT2_SUB_X)

    ADA2_AFT2_PRO_X = np.array(df["ADA2_AFT2_PRO"]).reshape(-1, 1)
    df["ADA2_AFT2_PRO"] = scaler.fit_transform(ADA2_AFT2_PRO_X)

    ADA2_PC1_SUM_X = np.array(df["ADA2_PC1_SUM"]).reshape(-1, 1)
    df["ADA2_PC1_SUM"] = scaler.fit_transform(ADA2_PC1_SUM_X)

    ADA2_PC1_SUB_X = np.array(df["ADA2_PC1_SUB"]).reshape(-1, 1)
    df["ADA2_PC1_SUB"] = scaler.fit_transform(ADA2_PC1_SUB_X)

    ADA2_PC1_PRO_X = np.array(df["ADA2_PC1_PRO"]).reshape(-1, 1)
    df["ADA2_PC1_PRO"] = scaler.fit_transform(ADA2_PC1_PRO_X)

    ADA2_PC2_SUM_X = np.array(df["ADA2_PC2_SUM"]).reshape(-1, 1)
    df["ADA2_PC2_SUM"] = scaler.fit_transform(ADA2_PC2_SUM_X)

    ADA2_PC2_SUB_X = np.array(df["ADA2_PC2_SUB"]).reshape(-1, 1)
    df["ADA2_PC2_SUB"] = scaler.fit_transform(ADA2_PC2_SUB_X)

    ADA2_PC2_PRO_X = np.array(df["ADA2_PC2_PRO"]).reshape(-1, 1)
    df["ADA2_PC2_PRO"] = scaler.fit_transform(ADA2_PC2_PRO_X)

    ADA2_PC3_SUM_X = np.array(df["ADA2_PC3_SUM"]).reshape(-1, 1)
    df["ADA2_PC3_SUM"] = scaler.fit_transform(ADA2_PC3_SUM_X)

    ADA2_PC3_SUB_X = np.array(df["ADA2_PC3_SUB"]).reshape(-1, 1)
    df["ADA2_PC3_SUB"] = scaler.fit_transform(ADA2_PC3_SUB_X)

    ADA2_PC3_PRO_X = np.array(df["ADA2_PC3_PRO"]).reshape(-1, 1)
    df["ADA2_PC3_PRO"] = scaler.fit_transform(ADA2_PC3_PRO_X)

    ADR1_AFT2_SUM_X = np.array(df["ADR1_AFT2_SUM"]).reshape(-1, 1)
    df["ADR1_AFT2_SUM"] = scaler.fit_transform(ADR1_AFT2_SUM_X)

    ADR1_AFT2_SUB_X = np.array(df["ADR1_AFT2_SUB"]).reshape(-1, 1)
    df["ADR1_AFT2_SUB"] = scaler.fit_transform(ADR1_AFT2_SUB_X)

    ADR1_AFT2_PRO_X = np.array(df["ADR1_AFT2_PRO"]).reshape(-1, 1)
    df["ADR1_AFT2_PRO"] = scaler.fit_transform(ADR1_AFT2_PRO_X)

    ADR1_PC1_SUM_X = np.array(df["ADR1_PC1_SUM"]).reshape(-1, 1)
    df["ADR1_PC1_SUM"] = scaler.fit_transform(ADR1_PC1_SUM_X)

    ADR1_PC1_SUB_X = np.array(df["ADR1_PC1_SUB"]).reshape(-1, 1)
    df["ADR1_PC1_SUB"] = scaler.fit_transform(ADR1_PC1_SUB_X)

    ADR1_PC1_PRO_X = np.array(df["ADR1_PC1_PRO"]).reshape(-1, 1)
    df["ADR1_PC1_PRO"] = scaler.fit_transform(ADR1_PC1_PRO_X)

    ADR1_PC2_SUM_X = np.array(df["ADR1_PC2_SUM"]).reshape(-1, 1)
    df["ADR1_PC2_SUM"] = scaler.fit_transform(ADR1_PC2_SUM_X)

    ADR1_PC2_SUB_X = np.array(df["ADR1_PC2_SUB"]).reshape(-1, 1)
    df["ADR1_PC2_SUB"] = scaler.fit_transform(ADR1_PC2_SUB_X)

    ADR1_PC2_PRO_X = np.array(df["ADR1_PC2_PRO"]).reshape(-1, 1)
    df["ADR1_PC2_PRO"] = scaler.fit_transform(ADR1_PC2_PRO_X)

    ADR1_PC3_SUM_X = np.array(df["ADR1_PC3_SUM"]).reshape(-1, 1)
    df["ADR1_PC3_SUM"] = scaler.fit_transform(ADR1_PC3_SUM_X)

    ADR1_PC3_SUB_X = np.array(df["ADR1_PC3_SUB"]).reshape(-1, 1)
    df["ADR1_PC3_SUB"] = scaler.fit_transform(ADR1_PC3_SUB_X)

    ADR1_PC3_PRO_X = np.array(df["ADR1_PC3_PRO"]).reshape(-1, 1)
    df["ADR1_PC3_PRO"] = scaler.fit_transform(ADR1_PC3_PRO_X)

    AFT2_PC1_SUM_X = np.array(df["AFT2_PC1_SUM"]).reshape(-1, 1)
    df["AFT2_PC1_SUM"] = scaler.fit_transform(AFT2_PC1_SUM_X)

    AFT2_PC1_SUB_X = np.array(df["AFT2_PC1_SUB"]).reshape(-1, 1)
    df["AFT2_PC1_SUB"] = scaler.fit_transform(AFT2_PC1_SUB_X)

    AFT2_PC1_PRO_X = np.array(df["AFT2_PC1_PRO"]).reshape(-1, 1)
    df["AFT2_PC1_PRO"] = scaler.fit_transform(AFT2_PC1_PRO_X)

    AFT2_PC2_SUM_X = np.array(df["AFT2_PC2_SUM"]).reshape(-1, 1)
    df["AFT2_PC2_SUM"] = scaler.fit_transform(AFT2_PC2_SUM_X)

    AFT2_PC2_SUB_X = np.array(df["AFT2_PC2_SUB"]).reshape(-1, 1)
    df["AFT2_PC2_SUB"] = scaler.fit_transform(AFT2_PC2_SUB_X)

    AFT2_PC2_PRO_X = np.array(df["AFT2_PC2_PRO"]).reshape(-1, 1)
    df["AFT2_PC2_PRO"] = scaler.fit_transform(AFT2_PC2_PRO_X)

    AFT2_PC3_SUM_X = np.array(df["AFT2_PC3_SUM"]).reshape(-1, 1)
    df["AFT2_PC3_SUM"] = scaler.fit_transform(AFT2_PC3_SUM_X)

    AFT2_PC3_SUB_X = np.array(df["AFT2_PC3_SUB"]).reshape(-1, 1)
    df["AFT2_PC3_SUB"] = scaler.fit_transform(AFT2_PC3_SUB_X)

    AFT2_PC3_PRO_X = np.array(df["AFT2_PC3_PRO"]).reshape(-1, 1)
    df["AFT2_PC3_PRO"] = scaler.fit_transform(AFT2_PC3_PRO_X)

    PC1_PC2_SUM_X = np.array(df["PC1_PC2_SUM"]).reshape(-1, 1)
    df["PC1_PC2_SUM"] = scaler.fit_transform(PC1_PC2_SUM_X)

    PC1_PC2_SUB_X = np.array(df["PC1_PC2_SUB"]).reshape(-1, 1)
    df["PC1_PC2_SUB"] = scaler.fit_transform(PC1_PC2_SUB_X)

    PC1_PC2_PRO_X = np.array(df["PC1_PC2_PRO"]).reshape(-1, 1)
    df["PC1_PC2_PRO"] = scaler.fit_transform(PC1_PC2_PRO_X)

    PC1_PC3_SUM_X = np.array(df["PC1_PC3_SUM"]).reshape(-1, 1)
    df["PC1_PC3_SUM"] = scaler.fit_transform(PC1_PC3_SUM_X)

    PC1_PC3_SUB_X = np.array(df["PC1_PC3_SUB"]).reshape(-1, 1)
    df["PC1_PC3_SUB"] = scaler.fit_transform(PC1_PC3_SUB_X)

    PC1_PC3_PRO_X = np.array(df["PC1_PC3_PRO"]).reshape(-1, 1)
    df["PC1_PC3_PRO"] = scaler.fit_transform(PC1_PC3_PRO_X)

    PC2_PC3_SUM_X = np.array(df["PC2_PC3_SUM"]).reshape(-1, 1)
    df["PC2_PC3_SUM"] = scaler.fit_transform(PC2_PC3_SUM_X)

    PC2_PC3_SUB_X = np.array(df["PC2_PC3_SUB"]).reshape(-1, 1)
    df["PC2_PC3_SUB"] = scaler.fit_transform(PC2_PC3_SUB_X)

    PC2_PC3_PRO_X = np.array(df["PC2_PC3_PRO"]).reshape(-1, 1)
    df["PC2_PC3_PRO"] = scaler.fit_transform(PC2_PC3_PRO_X)


    c = 0
    # Calcualte PE and KE of all initial molecules using SVM
    for row in range(TotalMolecule):

        features.clear()
        for j in range(287):
            if copy_molecules[row][j] == 1 and j == 0:
                features.append('SL')
            elif copy_molecules[row][j] == 1 and j == 1:
                features.append('DC')
            elif copy_molecules[row][j] == 1 and j == 2:
                features.append('EC')
            elif copy_molecules[row][j] == 1 and j == 3:
                features.append('IC')
            elif copy_molecules[row][j] == 1 and j == 4:
                features.append('LAC')
            elif copy_molecules[row][j] == 1 and j == 5:
                features.append('BC')
            elif copy_molecules[row][j] == 1 and j == 6:
                features.append('CC')
            elif copy_molecules[row][j] == 1 and j == 7:
                features.append('NC')
            elif copy_molecules[row][j] == 1 and j == 8:
                features.append('ADA2')
            elif copy_molecules[row][j] == 1 and j == 9:
                features.append('ADR1')
            elif copy_molecules[row][j] == 1 and j == 10:
                features.append('AFT2')
            elif copy_molecules[row][j] == 1 and j == 11:
                features.append('PC1')
            elif copy_molecules[row][j] == 1 and j == 12:
                features.append('PC2')
            elif copy_molecules[row][j] == 1 and j == 13:
                features.append('PC3')
            elif copy_molecules[row][j] == 1 and j == 14:
                features.append('SL_DC_SUM')
            elif copy_molecules[row][j] == 1 and j == 15:
                features.append('SL_DC_SUB')
            elif copy_molecules[row][j] == 1 and j == 16:
                features.append('SL_DC_PRO')
            elif copy_molecules[row][j] == 1 and j == 17:
                features.append('SL_EC_SUM')
            elif copy_molecules[row][j] == 1 and j == 18:
                features.append('SL_EC_SUB')
            elif copy_molecules[row][j] == 1 and j == 19:
                features.append('SL_EC_PRO')
            elif copy_molecules[row][j] == 1 and j == 20:
                features.append('SL_IC_SUM')
            elif copy_molecules[row][j] == 1 and j == 21:
                features.append('SL_IC_SUB')
            elif copy_molecules[row][j] == 1 and j == 22:
                features.append('SL_IC_PRO')
            elif copy_molecules[row][j] == 1 and j == 23:
                features.append('SL_LAC_SUM')
            elif copy_molecules[row][j] == 1 and j == 24:
                features.append('SL_LAC_SUB')
            elif copy_molecules[row][j] == 1 and j == 25:
                features.append('SL_LAC_PRO')
            elif copy_molecules[row][j] == 1 and j == 26:
                features.append('SL_BC_SUM')
            elif copy_molecules[row][j] == 1 and j == 27:
                features.append('SL_BC_SUB')
            elif copy_molecules[row][j] == 1 and j == 28:
                features.append('SL_BC_PRO')
            elif copy_molecules[row][j] == 1 and j == 29:
                features.append('SL_CC_SUM')
            elif copy_molecules[row][j] == 1 and j == 30:
                features.append('SL_CC_SUB')
            elif copy_molecules[row][j] == 1 and j == 31:
                features.append('SL_CC_PRO')
            elif copy_molecules[row][j] == 1 and j == 32:
                features.append('SL_NC_SUM')
            elif copy_molecules[row][j] == 1 and j == 33:
                features.append('SL_NC_SUB')
            elif copy_molecules[row][j] == 1 and j == 34:
                features.append('SL_NC_PRO')
            elif copy_molecules[row][j] == 1 and j == 35:
                features.append('SL_ADA2_SUM')
            elif copy_molecules[row][j] == 1 and j == 36:
                features.append('SL_ADA2_SUB')
            elif copy_molecules[row][j] == 1 and j == 37:
                features.append('SL_ADA2_PRO')
            elif copy_molecules[row][j] == 1 and j == 38:
                features.append('SL_ADR1_SUM')
            elif copy_molecules[row][j] == 1 and j == 39:
                features.append('SL_ADR1_SUB')
            elif copy_molecules[row][j] == 1 and j == 40:
                features.append('SL_ADR1_PRO')
            elif copy_molecules[row][j] == 1 and j == 41:
                features.append('SL_AFT2_SUM')
            elif copy_molecules[row][j] == 1 and j == 42:
                features.append('SL_AFT2_SUB')
            elif copy_molecules[row][j] == 1 and j == 43:
                features.append('SL_AFT2_PRO')
            elif copy_molecules[row][j] == 1 and j == 44:
                features.append('SL_PC1_SUM')
            elif copy_molecules[row][j] == 1 and j == 45:
                features.append('SL_PC1_SUB')
            elif copy_molecules[row][j] == 1 and j == 46:
                features.append('SL_PC1_PRO')
            elif copy_molecules[row][j] == 1 and j == 47:
                features.append('SL_PC2_SUM')
            elif copy_molecules[row][j] == 1 and j == 48:
                features.append('SL_PC2_SUB')
            elif copy_molecules[row][j] == 1 and j == 49:
                features.append('SL_PC2_PRO')
            elif copy_molecules[row][j] == 1 and j == 50:
                features.append('SL_PC3_SUM')
            elif copy_molecules[row][j] == 1 and j == 51:
                features.append('SL_PC3_SUB')
            elif copy_molecules[row][j] == 1 and j == 52:
                features.append('SL_PC3_PRO')
            elif copy_molecules[row][j] == 1 and j == 53:
                features.append('DC_EC_SUM')
            elif copy_molecules[row][j] == 1 and j == 54:
                features.append('DC_EC_SUB')
            elif copy_molecules[row][j] == 1 and j == 55:
                features.append('DC_EC_PRO')
            elif copy_molecules[row][j] == 1 and j == 56:
                features.append('DC_IC_SUM')
            elif copy_molecules[row][j] == 1 and j == 57:
                features.append('DC_IC_SUB')
            elif copy_molecules[row][j] == 1 and j == 58:
                features.append('DC_IC_PRO')
            elif copy_molecules[row][j] == 1 and j == 59:
                features.append('DC_LAC_SUM')
            elif copy_molecules[row][j] == 1 and j == 60:
                features.append('DC_LAC_SUB')
            elif copy_molecules[row][j] == 1 and j == 61:
                features.append('DC_LAC_PRO')
            elif copy_molecules[row][j] == 1 and j == 62:
                features.append('DC_BC_SUM')
            elif copy_molecules[row][j] == 1 and j == 63:
                features.append('DC_BC_SUB')
            elif copy_molecules[row][j] == 1 and j == 64:
                features.append('DC_BC_PRO')
            elif copy_molecules[row][j] == 1 and j == 65:
                features.append('DC_CC_SUM')
            elif copy_molecules[row][j] == 1 and j == 66:
                features.append('DC_CC_SUB')
            elif copy_molecules[row][j] == 1 and j == 67:
                features.append('DC_CC_PRO')
            elif copy_molecules[row][j] == 1 and j == 68:
                features.append('DC_NC_SUM')
            elif copy_molecules[row][j] == 1 and j == 69:
                features.append('DC_NC_SUB')
            elif copy_molecules[row][j] == 1 and j == 70:
                features.append('DC_NC_PRO')
            elif copy_molecules[row][j] == 1 and j == 71:
                features.append('DC_ADA2_SUM')
            elif copy_molecules[row][j] == 1 and j == 72:
                features.append('DC_ADA2_SUB')
            elif copy_molecules[row][j] == 1 and j == 73:
                features.append('DC_ADA2_PRO')
            elif copy_molecules[row][j] == 1 and j == 74:
                features.append('DC_ADR1_SUM')
            elif copy_molecules[row][j] == 1 and j == 75:
                features.append('DC_ADR1_SUB')
            elif copy_molecules[row][j] == 1 and j == 76:
                features.append('DC_ADR1_PRO')
            elif copy_molecules[row][j] == 1 and j == 77:
                features.append('DC_AFT2_SUM')
            elif copy_molecules[row][j] == 1 and j == 78:
                features.append('DC_AFT2_SUB')
            elif copy_molecules[row][j] == 1 and j == 79:
                features.append('DC_AFT2_PRO')
            elif copy_molecules[row][j] == 1 and j == 80:
                features.append('DC_PC1_SUM')
            elif copy_molecules[row][j] == 1 and j == 81:
                features.append('DC_PC1_SUB')
            elif copy_molecules[row][j] == 1 and j == 82:
                features.append('DC_PC1_PRO')
            elif copy_molecules[row][j] == 1 and j == 83:
                features.append('DC_PC2_SUM')
            elif copy_molecules[row][j] == 1 and j == 84:
                features.append('DC_PC2_SUB')
            elif copy_molecules[row][j] == 1 and j == 85:
                features.append('DC_PC2_PRO')
            elif copy_molecules[row][j] == 1 and j == 86:
                features.append('DC_PC3_SUM')
            elif copy_molecules[row][j] == 1 and j == 87:
                features.append('DC_PC3_SUB')
            elif copy_molecules[row][j] == 1 and j == 88:
                features.append('DC_PC3_PRO')
            elif copy_molecules[row][j] == 1 and j == 89:
                features.append('EC_IC_SUM')
            elif copy_molecules[row][j] == 1 and j == 90:
                features.append('EC_IC_SUB')
            elif copy_molecules[row][j] == 1 and j == 91:
                features.append('EC_IC_PRO')
            elif copy_molecules[row][j] == 1 and j == 92:
                features.append('EC_LAC_SUM')
            elif copy_molecules[row][j] == 1 and j == 93:
                features.append('EC_LAC_SUB')
            elif copy_molecules[row][j] == 1 and j == 94:
                features.append('EC_LAC_PRO')
            elif copy_molecules[row][j] == 1 and j == 95:
                features.append('EC_BC_SUM')
            elif copy_molecules[row][j] == 1 and j == 96:
                features.append('EC_BC_SUB')
            elif copy_molecules[row][j] == 1 and j == 97:
                features.append('EC_BC_PRO')
            elif copy_molecules[row][j] == 1 and j == 98:
                features.append('EC_CC_SUM')
            elif copy_molecules[row][j] == 1 and j == 99:
                features.append('EC_CC_SUB')
            elif copy_molecules[row][j] == 1 and j == 100:
                features.append('EC_CC_PRO')
            elif copy_molecules[row][j] == 1 and j == 101:
                features.append('EC_NC_SUM')
            elif copy_molecules[row][j] == 1 and j == 102:
                features.append('EC_NC_SUB')
            elif copy_molecules[row][j] == 1 and j == 103:
                features.append('EC_NC_PRO')
            elif copy_molecules[row][j] == 1 and j == 104:
                features.append('EC_ADA2_SUM')
            elif copy_molecules[row][j] == 1 and j == 105:
                features.append('EC_ADA2_SUB')
            elif copy_molecules[row][j] == 1 and j == 106:
                features.append('EC_ADA2_PRO')
            elif copy_molecules[row][j] == 1 and j == 107:
                features.append('EC_ADR1_SUM')
            elif copy_molecules[row][j] == 1 and j == 108:
                features.append('EC_ADR1_SUB')
            elif copy_molecules[row][j] == 1 and j == 109:
                features.append('EC_ADR1_PRO')
            elif copy_molecules[row][j] == 1 and j == 110:
                features.append('EC_AFT2_SUM')
            elif copy_molecules[row][j] == 1 and j == 111:
                features.append('EC_AFT2_SUB')
            elif copy_molecules[row][j] == 1 and j == 112:
                features.append('EC_AFT2_PRO')
            elif copy_molecules[row][j] == 1 and j == 113:
                features.append('EC_PC1_SUM')
            elif copy_molecules[row][j] == 1 and j == 114:
                features.append('EC_PC1_SUB')
            elif copy_molecules[row][j] == 1 and j == 115:
                features.append('EC_PC1_PRO')
            elif copy_molecules[row][j] == 1 and j == 116:
                features.append('EC_PC2_SUM')
            elif copy_molecules[row][j] == 1 and j == 117:
                features.append('EC_PC2_SUB')
            elif copy_molecules[row][j] == 1 and j == 118:
                features.append('EC_PC2_PRO')
            elif copy_molecules[row][j] == 1 and j == 119:
                features.append('EC_PC3_SUM')
            elif copy_molecules[row][j] == 1 and j == 120:
                features.append('EC_PC3_SUB')
            elif copy_molecules[row][j] == 1 and j == 121:
                features.append('EC_PC3_PRO')
            elif copy_molecules[row][j] == 1 and j == 122:
                features.append('IC_LAC_SUM')
            elif copy_molecules[row][j] == 1 and j == 123:
                features.append('IC_LAC_SUB')
            elif copy_molecules[row][j] == 1 and j == 124:
                features.append('IC_LAC_PRO')
            elif copy_molecules[row][j] == 1 and j == 125:
                features.append('IC_BC_SUM')
            elif copy_molecules[row][j] == 1 and j == 126:
                features.append('IC_BC_SUB')
            elif copy_molecules[row][j] == 1 and j == 127:
                features.append('IC_BC_PRO')
            elif copy_molecules[row][j] == 1 and j == 128:
                features.append('IC_CC_SUM')
            elif copy_molecules[row][j] == 1 and j == 129:
                features.append('IC_CC_SUB')
            elif copy_molecules[row][j] == 1 and j == 130:
                features.append('IC_CC_PRO')
            elif copy_molecules[row][j] == 1 and j == 131:
                features.append('IC_NC_SUM')
            elif copy_molecules[row][j] == 1 and j == 132:
                features.append('IC_NC_SUB')
            elif copy_molecules[row][j] == 1 and j == 133:
                features.append('IC_NC_PRO')
            elif copy_molecules[row][j] == 1 and j == 134:
                features.append('IC_ADA2_SUM')
            elif copy_molecules[row][j] == 1 and j == 135:
                features.append('IC_ADA2_SUB')
            elif copy_molecules[row][j] == 1 and j == 136:
                features.append('IC_ADA2_PRO')
            elif copy_molecules[row][j] == 1 and j == 137:
                features.append('IC_ADR1_SUM')
            elif copy_molecules[row][j] == 1 and j == 138:
                features.append('IC_ADR1_SUB')
            elif copy_molecules[row][j] == 1 and j == 139:
                features.append('IC_ADR1_PRO')
            elif copy_molecules[row][j] == 1 and j == 140:
                features.append('IC_AFT2_SUM')
            elif copy_molecules[row][j] == 1 and j == 141:
                features.append('IC_AFT2_SUB')
            elif copy_molecules[row][j] == 1 and j == 142:
                features.append('IC_AFT2_PRO')
            elif copy_molecules[row][j] == 1 and j == 143:
                features.append('IC_PC1_SUM')
            elif copy_molecules[row][j] == 1 and j == 144:
                features.append('IC_PC1_SUB')
            elif copy_molecules[row][j] == 1 and j == 145:
                features.append('IC_PC1_PRO')
            elif copy_molecules[row][j] == 1 and j == 146:
                features.append('IC_PC2_SUM')
            elif copy_molecules[row][j] == 1 and j == 147:
                features.append('IC_PC2_SUB')
            elif copy_molecules[row][j] == 1 and j == 148:
                features.append('IC_PC2_PRO')
            elif copy_molecules[row][j] == 1 and j == 149:
                features.append('IC_PC3_SUM')
            elif copy_molecules[row][j] == 1 and j == 150:
                features.append('IC_PC3_SUB')
            elif copy_molecules[row][j] == 1 and j == 151:
                features.append('IC_PC3_PRO')
            elif copy_molecules[row][j] == 1 and j == 152:
                features.append('LAC_BC_SUM')
            elif copy_molecules[row][j] == 1 and j == 153:
                features.append('LAC_BC_SUB')
            elif copy_molecules[row][j] == 1 and j == 154:
                features.append('LAC_BC_PRO')
            elif copy_molecules[row][j] == 1 and j == 155:
                features.append('LAC_CC_SUM')
            elif copy_molecules[row][j] == 1 and j == 156:
                features.append('LAC_CC_SUB')
            elif copy_molecules[row][j] == 1 and j == 157:
                features.append('LAC_CC_PRO')
            elif copy_molecules[row][j] == 1 and j == 158:
                features.append('LAC_NC_SUM')
            elif copy_molecules[row][j] == 1 and j == 159:
                features.append('LAC_NC_SUB')
            elif copy_molecules[row][j] == 1 and j == 160:
                features.append('LAC_NC_PRO')
            elif copy_molecules[row][j] == 1 and j == 161:
                features.append('LAC_ADA2_SUM')
            elif copy_molecules[row][j] == 1 and j == 162:
                features.append('LAC_ADA2_SUB')
            elif copy_molecules[row][j] == 1 and j == 163:
                features.append('LAC_ADA2_PRO')
            elif copy_molecules[row][j] == 1 and j == 164:
                features.append('LAC_ADR1_SUM')
            elif copy_molecules[row][j] == 1 and j == 165:
                features.append('LAC_ADR1_SUB')
            elif copy_molecules[row][j] == 1 and j == 166:
                features.append('LAC_ADR1_PRO')
            elif copy_molecules[row][j] == 1 and j == 167:
                features.append('LAC_AFT2_SUM')
            elif copy_molecules[row][j] == 1 and j == 168:
                features.append('LAC_AFT2_SUB')
            elif copy_molecules[row][j] == 1 and j == 169:
                features.append('LAC_AFT2_PRO')
            elif copy_molecules[row][j] == 1 and j == 170:
                features.append('LAC_PC1_SUM')
            elif copy_molecules[row][j] == 1 and j == 171:
                features.append('LAC_PC1_SUB')
            elif copy_molecules[row][j] == 1 and j == 172:
                features.append('LAC_PC1_PRO')
            elif copy_molecules[row][j] == 1 and j == 173:
                features.append('LAC_PC2_SUM')
            elif copy_molecules[row][j] == 1 and j == 174:
                features.append('LAC_PC2_SUB')
            elif copy_molecules[row][j] == 1 and j == 175:
                features.append('LAC_PC2_PRO')
            elif copy_molecules[row][j] == 1 and j == 176:
                features.append('LAC_PC3_SUM')
            elif copy_molecules[row][j] == 1 and j == 177:
                features.append('LAC_PC3_SUB')
            elif copy_molecules[row][j] == 1 and j == 178:
                features.append('LAC_PC3_PRO')
            elif copy_molecules[row][j] == 1 and j == 179:
                features.append('BC_CC_SUM')
            elif copy_molecules[row][j] == 1 and j == 180:
                features.append('BC_CC_SUB')
            elif copy_molecules[row][j] == 1 and j == 181:
                features.append('BC_CC_PRO')
            elif copy_molecules[row][j] == 1 and j == 182:
                features.append('BC_NC_SUM')
            elif copy_molecules[row][j] == 1 and j == 183:
                features.append('BC_NC_SUB')
            elif copy_molecules[row][j] == 1 and j == 184:
                features.append('BC_NC_PRO')
            elif copy_molecules[row][j] == 1 and j == 185:
                features.append('BC_ADA2_SUM')
            elif copy_molecules[row][j] == 1 and j == 186:
                features.append('BC_ADA2_SUB')
            elif copy_molecules[row][j] == 1 and j == 187:
                features.append('BC_ADA2_PRO')
            elif copy_molecules[row][j] == 1 and j == 188:
                features.append('BC_ADR1_SUM')
            elif copy_molecules[row][j] == 1 and j == 189:
                features.append('BC_ADR1_SUB')
            elif copy_molecules[row][j] == 1 and j == 190:
                features.append('BC_ADR1_PRO')
            elif copy_molecules[row][j] == 1 and j == 191:
                features.append('BC_AFT2_SUM')
            elif copy_molecules[row][j] == 1 and j == 192:
                features.append('BC_AFT2_SUB')
            elif copy_molecules[row][j] == 1 and j == 193:
                features.append('BC_AFT2_PRO')
            elif copy_molecules[row][j] == 1 and j == 194:
                features.append('BC_PC1_SUM')
            elif copy_molecules[row][j] == 1 and j == 195:
                features.append('BC_PC1_SUB')
            elif copy_molecules[row][j] == 1 and j == 196:
                features.append('BC_PC1_PRO')
            elif copy_molecules[row][j] == 1 and j == 197:
                features.append('BC_PC2_SUM')
            elif copy_molecules[row][j] == 1 and j == 198:
                features.append('BC_PC2_SUB')
            elif copy_molecules[row][j] == 1 and j == 199:
                features.append('BC_PC2_PRO')
            elif copy_molecules[row][j] == 1 and j == 200:
                features.append('BC_PC3_SUM')
            elif copy_molecules[row][j] == 1 and j == 201:
                features.append('BC_PC3_SUB')
            elif copy_molecules[row][j] == 1 and j == 202:
                features.append('BC_PC3_PRO')
            elif copy_molecules[row][j] == 1 and j == 203:
                features.append('CC_NC_SUM')
            elif copy_molecules[row][j] == 1 and j == 204:
                features.append('CC_NC_SUB')
            elif copy_molecules[row][j] == 1 and j == 205:
                features.append('CC_NC_PRO')
            elif copy_molecules[row][j] == 1 and j == 206:
                features.append('CC_ADA2_SUM')
            elif copy_molecules[row][j] == 1 and j == 207:
                features.append('CC_ADA2_SUB')
            elif copy_molecules[row][j] == 1 and j == 208:
                features.append('CC_ADA2_PRO')
            elif copy_molecules[row][j] == 1 and j == 209:
                features.append('CC_ADR1_SUM')
            elif copy_molecules[row][j] == 1 and j == 210:
                features.append('CC_ADR1_SUB')
            elif copy_molecules[row][j] == 1 and j == 211:
                features.append('CC_ADR1_PRO')
            elif copy_molecules[row][j] == 1 and j == 212:
                features.append('CC_AFT2_SUM')
            elif copy_molecules[row][j] == 1 and j == 213:
                features.append('CC_AFT2_SUB')
            elif copy_molecules[row][j] == 1 and j == 214:
                features.append('CC_AFT2_PRO')
            elif copy_molecules[row][j] == 1 and j == 215:
                features.append('CC_PC1_SUM')
            elif copy_molecules[row][j] == 1 and j == 216:
                features.append('CC_PC1_SUB')
            elif copy_molecules[row][j] == 1 and j == 217:
                features.append('CC_PC1_PRO')
            elif copy_molecules[row][j] == 1 and j == 218:
                features.append('CC_PC2_SUM')
            elif copy_molecules[row][j] == 1 and j == 219:
                features.append('CC_PC2_SUB')
            elif copy_molecules[row][j] == 1 and j == 220:
                features.append('CC_PC2_PRO')
            elif copy_molecules[row][j] == 1 and j == 221:
                features.append('CC_PC3_SUM')
            elif copy_molecules[row][j] == 1 and j == 222:
                features.append('CC_PC3_SUB')
            elif copy_molecules[row][j] == 1 and j == 223:
                features.append('CC_PC3_PRO')
            elif copy_molecules[row][j] == 1 and j == 224:
                features.append('NC_ADA2_SUM')
            elif copy_molecules[row][j] == 1 and j == 225:
                features.append('NC_ADA2_SUB')
            elif copy_molecules[row][j] == 1 and j == 226:
                features.append('NC_ADA2_PRO')
            elif copy_molecules[row][j] == 1 and j == 227:
                features.append('NC_ADR1_SUM')
            elif copy_molecules[row][j] == 1 and j == 228:
                features.append('NC_ADR1_SUB')
            elif copy_molecules[row][j] == 1 and j == 229:
                features.append('NC_ADR1_PRO')
            elif copy_molecules[row][j] == 1 and j == 230:
                features.append('NC_AFT2_SUM')
            elif copy_molecules[row][j] == 1 and j == 231:
                features.append('NC_AFT2_SUB')
            elif copy_molecules[row][j] == 1 and j == 232:
                features.append('NC_AFT2_PRO')
            elif copy_molecules[row][j] == 1 and j == 233:
                features.append('NC_PC1_SUM')
            elif copy_molecules[row][j] == 1 and j == 234:
                features.append('NC_PC1_SUB')
            elif copy_molecules[row][j] == 1 and j == 235:
                features.append('NC_PC1_PRO')
            elif copy_molecules[row][j] == 1 and j == 236:
                features.append('NC_PC2_SUM')
            elif copy_molecules[row][j] == 1 and j == 237:
                features.append('NC_PC2_SUB')
            elif copy_molecules[row][j] == 1 and j == 238:
                features.append('NC_PC2_PRO')
            elif copy_molecules[row][j] == 1 and j == 239:
                features.append('NC_PC3_SUM')
            elif copy_molecules[row][j] == 1 and j == 240:
                features.append('NC_PC3_SUB')
            elif copy_molecules[row][j] == 1 and j == 241:
                features.append('NC_PC3_PRO')
            elif copy_molecules[row][j] == 1 and j == 242:
                features.append('ADA2_ADR1_SUM')
            elif copy_molecules[row][j] == 1 and j == 243:
                features.append('ADA2_ADR1_SUB')
            elif copy_molecules[row][j] == 1 and j == 244:
                features.append('ADA2_ADR1_PRO')
            elif copy_molecules[row][j] == 1 and j == 245:
                features.append('ADA2_AFT2_SUM')
            elif copy_molecules[row][j] == 1 and j == 246:
                features.append('ADA2_AFT2_SUB')
            elif copy_molecules[row][j] == 1 and j == 247:
                features.append('ADA2_AFT2_PRO')
            elif copy_molecules[row][j] == 1 and j == 248:
                features.append('ADA2_PC1_SUM')
            elif copy_molecules[row][j] == 1 and j == 249:
                features.append('ADA2_PC1_SUB')
            elif copy_molecules[row][j] == 1 and j == 250:
                features.append('ADA2_PC1_PRO')
            elif copy_molecules[row][j] == 1 and j == 251:
                features.append('ADA2_PC2_SUM')
            elif copy_molecules[row][j] == 1 and j == 252:
                features.append('ADA2_PC2_SUB')
            elif copy_molecules[row][j] == 1 and j == 253:
                features.append('ADA2_PC2_PRO')
            elif copy_molecules[row][j] == 1 and j == 254:
                features.append('ADA2_PC3_SUM')
            elif copy_molecules[row][j] == 1 and j == 255:
                features.append('ADA2_PC3_SUB')
            elif copy_molecules[row][j] == 1 and j == 256:
                features.append('ADA2_PC3_PRO')
            elif copy_molecules[row][j] == 1 and j == 257:
                features.append('ADR1_AFT2_SUM')
            elif copy_molecules[row][j] == 1 and j == 258:
                features.append('ADR1_AFT2_SUB')
            elif copy_molecules[row][j] == 1 and j == 259:
                features.append('ADR1_AFT2_PRO')
            elif copy_molecules[row][j] == 1 and j == 260:
                features.append('ADR1_PC1_SUM')
            elif copy_molecules[row][j] == 1 and j == 261:
                features.append('ADR1_PC1_SUB')
            elif copy_molecules[row][j] == 1 and j == 262:
                features.append('ADR1_PC1_PRO')
            elif copy_molecules[row][j] == 1 and j == 263:
                features.append('ADR1_PC2_SUM')
            elif copy_molecules[row][j] == 1 and j == 264:
                features.append('ADR1_PC2_SUB')
            elif copy_molecules[row][j] == 1 and j == 265:
                features.append('ADR1_PC2_PRO')
            elif copy_molecules[row][j] == 1 and j == 266:
                features.append('ADR1_PC3_SUM')
            elif copy_molecules[row][j] == 1 and j == 267:
                features.append('ADR1_PC3_SUB')
            elif copy_molecules[row][j] == 1 and j == 268:
                features.append('ADR1_PC3_PRO')
            elif copy_molecules[row][j] == 1 and j == 269:
                features.append('AFT2_PC1_SUM')
            elif copy_molecules[row][j] == 1 and j == 270:
                features.append('AFT2_PC1_SUB')
            elif copy_molecules[row][j] == 1 and j == 271:
                features.append('AFT2_PC1_PRO')
            elif copy_molecules[row][j] == 1 and j == 272:
                features.append('AFT2_PC2_SUM')
            elif copy_molecules[row][j] == 1 and j == 273:
                features.append('AFT2_PC2_SUB')
            elif copy_molecules[row][j] == 1 and j == 274:
                features.append('AFT2_PC2_PRO')
            elif copy_molecules[row][j] == 1 and j == 275:
                features.append('AFT2_PC3_SUM')
            elif copy_molecules[row][j] == 1 and j == 276:
                features.append('AFT2_PC3_SUB')
            elif copy_molecules[row][j] == 1 and j == 277:
                features.append('AFT2_PC3_PRO')
            elif copy_molecules[row][j] == 1 and j == 278:
                features.append('PC1_PC2_SUM')
            elif copy_molecules[row][j] == 1 and j == 279:
                features.append('PC1_PC2_SUB')
            elif copy_molecules[row][j] == 1 and j == 280:
                features.append('PC1_PC2_PRO')
            elif copy_molecules[row][j] == 1 and j == 281:
                features.append('PC1_PC3_SUM')
            elif copy_molecules[row][j] == 1 and j == 282:
                features.append('PC1_PC3_SUB')
            elif copy_molecules[row][j] == 1 and j == 283:
                features.append('PC1_PC3_PRO')
            elif copy_molecules[row][j] == 1 and j == 284:
                features.append('PC2_PC3_SUM')
            elif copy_molecules[row][j] == 1 and j == 285:
                features.append('PC2_PC3_SUB')
            elif copy_molecules[row][j] == 1 and j == 286:
                features.append('PC2_PC3_PRO')

        similar = 0
        for j in range(287):
            if copy_molecules[row][j] == 0:
                similar = similar + 1
        c = c + 1
        print('{}. Current Molecule: {}    Current Features: {}'.format(c,copy_molecules[row], features))

        # read dataset
        n = TotalMolecule
        X = df[features]
        y = df['P_Status']
        # Make comment line 1590, 1591, and 1592  to run this program using the imbalance data. Then Remove comment line from 1593.
        smote = SMOTEENN(sampling_strategy='minority')
        X_smote, y_smote = smote.fit_resample(X, y)
        cal_accuracy(self, X_smote, y_smote)
        #cal_accuracy(self, X, y)

    CRI_iterations(self)

def cal_accuracy(self, X, y):
    global MinPE, MinStruct, NumHit, PE, KE

    # Remove comment form 1601 to use XGBoost Classifier in this method. Make comment from line 1613 to 1626.
    #clf = XGBClassifier()
    # Remove comment form 1603 to use LightGBM Classifier in this method. Make comment from line 1613 to 1626.
    #clf = lgb.LGBMClassifier()
    # Remove comment form 1605 to use Random Forest Classifier in this method. Make comment from line 1613 to 1626.
    clf = RandomForestClassifier(n_estimators=100)

    # Remove comment form line 1608 to line 1610 to use either XGBoost, LightGBM, or Random Forest Classifier in this method. Make comment from line 1613 to 1626.
    scores = cross_val_score(estimator=clf, X=X, y=y, cv=10, scoring='accuracy')
    predicted_label = cross_val_predict(estimator=clf, X=X, y=y, cv=10)
    score = round(scores.mean(), 6)

    # Remove Comment lines from 1613 to 1626 to execute the method using the ensemble method. Make comment all lines from 1601 to 1610
    #estimators = []
    #estimators.clear()
    #clf = XGBClassifier(max_depth=4, scale_pos_weight=3)
    #estimators.append(('XGBoost', clf))

    #clf2 = RandomForestClassifier(n_estimators=100)
    #estimators.append(('RandomForest', clf2))

    #clf3 = lgb.LGBMClassifier()
    #estimators.append(('LightGBM', clf3))

    #ensemble = VotingClassifier(estimators)
    #scores = cross_val_score(estimator=ensemble, X=X, y=y, cv=10, scoring='accuracy')
    #score = round(scores.mean(), 6)

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

def objective_unction_gain_check(self,mol_features,child,obj_func_gain):
    n = 0
    N=287
    for j in range(287):
        if mol_features[child][j] == 1:
            n = n + 1
    new_obj_func_gain = (1-(obj_func_gain/100))-((c*N)/n)
    print('n= {} Accuracy_Received= {} New_obj_Func_achieved= {}'.format(n,obj_func_gain,new_obj_func_gain))
    return new_obj_func_gain

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
    Min_obj_func_val= 100
    no_of_iteration = 0
    while terminate < 1000:
        no_of_iteration = no_of_iteration + 1
        print('Iteration number: {}'.format(no_of_iteration))
        t = random.uniform(0, 1)
        # Decides whether to perform uni-molecular or inter-molecular reaction will occur
        if t > MoleColl:
            alpha = random.randint(0, 40)
            # Randomly Select a Molecule for Uni Molecular Reaction
            child = random.randint(0, TotalMolecule - 1)

            if (T_NumHit - T_MinHit) > alpha:
                # New Child Molecule 1
                print('                         Decomposition')
                count_decom = count_decom + 1
                copy_molecules = molecule.copy()
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
                copy_molecules[child][184] = 1
                copy_molecules[child][185] = 1
                copy_molecules[child][186] = 1
                copy_molecules[child][187] = 1
                copy_molecules[child][188] = 1
                copy_molecules[child][189] = 1
                copy_molecules[child][190] = 1
                copy_molecules[child][191] = 1
                copy_molecules[child][192] = 1
                copy_molecules[child][193] = 1
                copy_molecules[child][194] = 1
                copy_molecules[child][195] = 1
                copy_molecules[child][196] = 1
                copy_molecules[child][197] = 1
                copy_molecules[child][198] = 1
                copy_molecules[child][199] = 1
                copy_molecules[child][200] = 1
                copy_molecules[child][201] = 1
                copy_molecules[child][202] = 1
                copy_molecules[child][203] = 1
                copy_molecules[child][204] = 1
                copy_molecules[child][205] = 1
                copy_molecules[child][206] = 1
                copy_molecules[child][207] = 1
                copy_molecules[child][208] = 1
                copy_molecules[child][209] = 1
                copy_molecules[child][210] = 1
                copy_molecules[child][211] = 1
                copy_molecules[child][212] = 1
                copy_molecules[child][213] = 1
                copy_molecules[child][214] = 1
                copy_molecules[child][215] = 1
                copy_molecules[child][216] = 1
                copy_molecules[child][217] = 1
                copy_molecules[child][218] = 1
                copy_molecules[child][219] = 1
                copy_molecules[child][220] = 1
                copy_molecules[child][221] = 1
                copy_molecules[child][222] = 1
                copy_molecules[child][223] = 1
                copy_molecules[child][224] = 1
                copy_molecules[child][225] = 1
                copy_molecules[child][226] = 1
                copy_molecules[child][227] = 1
                copy_molecules[child][228] = 1
                copy_molecules[child][229] = 1
                copy_molecules[child][230] = 1
                copy_molecules[child][231] = 1
                copy_molecules[child][232] = 1
                copy_molecules[child][233] = 1
                copy_molecules[child][234] = 1
                copy_molecules[child][235] = 1
                copy_molecules[child][236] = 1
                copy_molecules[child][237] = 1
                copy_molecules[child][238] = 1
                copy_molecules[child][239] = 1
                copy_molecules[child][240] = 1
                copy_molecules[child][241] = 1
                copy_molecules[child][242] = 1
                copy_molecules[child][243] = 1
                copy_molecules[child][244] = 1
                copy_molecules[child][245] = 1
                copy_molecules[child][246] = 1
                copy_molecules[child][247] = 1
                copy_molecules[child][248] = 1
                copy_molecules[child][249] = 1
                copy_molecules[child][250] = 1
                copy_molecules[child][251] = 1
                copy_molecules[child][252] = 1
                copy_molecules[child][253] = 1
                copy_molecules[child][254] = 1
                copy_molecules[child][255] = 1
                copy_molecules[child][256] = 1
                copy_molecules[child][257] = 1
                copy_molecules[child][258] = 1
                copy_molecules[child][259] = 1
                copy_molecules[child][260] = 1
                copy_molecules[child][261] = 1
                copy_molecules[child][262] = 1
                copy_molecules[child][263] = 1
                copy_molecules[child][264] = 1
                copy_molecules[child][265] = 1
                copy_molecules[child][266] = 1
                copy_molecules[child][267] = 1
                copy_molecules[child][268] = 1
                copy_molecules[child][269] = 1
                copy_molecules[child][270] = 1
                copy_molecules[child][271] = 1
                copy_molecules[child][272] = 1
                copy_molecules[child][273] = 1
                copy_molecules[child][274] = 1
                copy_molecules[child][275] = 1
                copy_molecules[child][276] = 1
                copy_molecules[child][277] = 1
                copy_molecules[child][278] = 1
                copy_molecules[child][279] = 1
                copy_molecules[child][280] = 1
                copy_molecules[child][281] = 1
                copy_molecules[child][282] = 1
                copy_molecules[child][283] = 1
                copy_molecules[child][284] = 1
                copy_molecules[child][285] = 1
                copy_molecules[child][286] = 1

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
                copy_molecules2[child][92] = 1
                copy_molecules2[child][93] = 1
                copy_molecules2[child][94] = 1
                copy_molecules2[child][95] = 1
                copy_molecules2[child][96] = 1
                copy_molecules2[child][97] = 1
                copy_molecules2[child][98] = 1
                copy_molecules2[child][99] = 1
                copy_molecules2[child][100] = 1
                copy_molecules2[child][101] = 1
                copy_molecules2[child][102] = 1
                copy_molecules2[child][103] = 1
                copy_molecules2[child][104] = 1
                copy_molecules2[child][105] = 1
                copy_molecules2[child][106] = 1
                copy_molecules2[child][107] = 1
                copy_molecules2[child][108] = 1
                copy_molecules2[child][109] = 1
                copy_molecules2[child][110] = 1
                copy_molecules2[child][111] = 1
                copy_molecules2[child][112] = 1
                copy_molecules2[child][113] = 1
                copy_molecules2[child][114] = 1
                copy_molecules2[child][115] = 1
                copy_molecules2[child][116] = 1
                copy_molecules2[child][117] = 1
                copy_molecules2[child][118] = 1
                copy_molecules2[child][119] = 1
                copy_molecules2[child][120] = 1
                copy_molecules2[child][121] = 1
                copy_molecules2[child][122] = 1
                copy_molecules2[child][123] = 1
                copy_molecules2[child][124] = 1
                copy_molecules2[child][125] = 1
                copy_molecules2[child][126] = 1
                copy_molecules2[child][127] = 1
                copy_molecules2[child][128] = 1
                copy_molecules2[child][129] = 1
                copy_molecules2[child][130] = 1
                copy_molecules2[child][131] = 1
                copy_molecules2[child][132] = 1
                copy_molecules2[child][133] = 1
                copy_molecules2[child][134] = 1
                copy_molecules2[child][135] = 1
                copy_molecules2[child][136] = 1
                copy_molecules2[child][137] = 1
                copy_molecules2[child][138] = 1
                copy_molecules2[child][139] = 1
                copy_molecules2[child][140] = 1
                copy_molecules2[child][141] = 1
                copy_molecules2[child][142] = 1
                copy_molecules2[child][143] = 1
                copy_molecules2[child][144] = 1

                print('Child1: {} Child2: {}'.format(copy_molecules[child], copy_molecules2[child]))
                acc_result_child_1 = cal_child_accuracy(self, copy_molecules, child)

                child1_obj_func_gain = objective_unction_gain_check(self,copy_molecules,child,acc_result_child_1)
                acc_result_child_2 = cal_child_accuracy(self, copy_molecules2, child)

                child2_obj_func_gain = objective_unction_gain_check(self,copy_molecules2,child,acc_result_child_2)
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
                    print('Child 1 objective gain = {} Min objective gain = {}'.format(child1_obj_func_gain,Min_obj_func_val))
                    print('Child 2 objective gain = {} Min objective gain = {}'.format(child2_obj_func_gain,Min_obj_func_val))
                    if child1_obj_func_gain < Min_obj_func_val:
                        print('Decom Got Imptovement. Child Acc = {} MinPE = {} and P_MinPE = {}: '.format(acc_result_child_1, MinPE, P_MinPE))
                        MinPE = acc_result_child_1
                        Min_obj_func_val = child1_obj_func_gain
                        MinStruct = copy_molecules[child].copy()
                    elif child2_obj_func_gain < Min_obj_func_val:
                        print('Decom Got Imptovement. Child Acc = {} MinPE = {} and P_MinPE = {}: '.format(acc_result_child_2, MinPE, P_MinPE))
                        MinPE = acc_result_child_2
                        Min_obj_func_val = child2_obj_func_gain
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
                rand_position = random.randint(0, 286)

                if copy_molecules[child][rand_position] == 0:
                    copy_molecules[child][rand_position] = 1
                else:
                    copy_molecules[child][rand_position] = 0
                # Repair Function Reject zero or one feature
                similar = 0
                for j in range(287):
                    # for j in range(7):
                    if copy_molecules[child][j] == 0:
                        similar = similar + 1
                if similar >= 286:
                    continue
                print('Child: {} '.format(copy_molecules[child]))
                acc_result_child = cal_child_accuracy(self, copy_molecules, child)

                child1_obj_func_gain = objective_unction_gain_check(self,copy_molecules,child,acc_result_child)
                acc_result_parent = PE[child]
                if acc_result_child < acc_result_parent + KE[child]:
                    alpha1 = random.uniform(KELossRate, 1)
                    new_KE = (acc_result_parent - acc_result_child + KE[child]) * alpha1
                    buffer = buffer + (acc_result_parent - acc_result_child + KE[child]) * (1 - alpha1)
                    PE[child] = acc_result_child
                    KE[child] = new_KE
                    NumHit[child] = NumHit[child] + 1
                    molecule[child][rand_position] = copy_molecules[child][rand_position]
                    print('Child objective gain = {} Min objective gain = {}'.format(child1_obj_func_gain,Min_obj_func_val))
                    if Min_obj_func_val > child1_obj_func_gain:
                        print('On Wall Got Imptovement. Child Acc = {} MinPE = {} and P_MinPE = {}: '.format(acc_result_child, MinPE, P_MinPE))
                        MinPE = acc_result_child
                        Min_obj_func_val = child1_obj_func_gain
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
                copy_molecules[random_child1][184] = molecule[random_child2][184]
                copy_molecules[random_child1][185] = molecule[random_child2][185]
                copy_molecules[random_child1][186] = molecule[random_child2][186]
                copy_molecules[random_child1][187] = molecule[random_child2][187]
                copy_molecules[random_child1][188] = molecule[random_child2][188]
                copy_molecules[random_child1][189] = molecule[random_child2][189]
                copy_molecules[random_child1][190] = molecule[random_child2][190]
                copy_molecules[random_child1][191] = molecule[random_child2][191]
                copy_molecules[random_child1][192] = molecule[random_child2][192]
                copy_molecules[random_child1][193] = molecule[random_child2][193]
                copy_molecules[random_child1][194] = molecule[random_child2][194]
                copy_molecules[random_child1][195] = molecule[random_child2][195]
                copy_molecules[random_child1][196] = molecule[random_child2][196]
                copy_molecules[random_child1][197] = molecule[random_child2][197]
                copy_molecules[random_child1][198] = molecule[random_child2][198]
                copy_molecules[random_child1][199] = molecule[random_child2][199]
                copy_molecules[random_child1][200] = molecule[random_child2][200]
                copy_molecules[random_child1][201] = molecule[random_child2][201]
                copy_molecules[random_child1][202] = molecule[random_child2][202]
                copy_molecules[random_child1][203] = molecule[random_child2][203]
                copy_molecules[random_child1][204] = molecule[random_child2][204]
                copy_molecules[random_child1][205] = molecule[random_child2][205]
                copy_molecules[random_child1][206] = molecule[random_child2][206]
                copy_molecules[random_child1][207] = molecule[random_child2][207]
                copy_molecules[random_child1][208] = molecule[random_child2][208]
                copy_molecules[random_child1][209] = molecule[random_child2][209]
                copy_molecules[random_child1][210] = molecule[random_child2][210]
                copy_molecules[random_child1][211] = molecule[random_child2][211]
                copy_molecules[random_child1][212] = molecule[random_child2][212]
                copy_molecules[random_child1][213] = molecule[random_child2][213]
                copy_molecules[random_child1][214] = molecule[random_child2][214]
                copy_molecules[random_child1][215] = molecule[random_child2][215]
                copy_molecules[random_child1][216] = molecule[random_child2][216]
                copy_molecules[random_child1][217] = molecule[random_child2][217]
                copy_molecules[random_child1][218] = molecule[random_child2][218]
                copy_molecules[random_child1][219] = molecule[random_child2][219]
                copy_molecules[random_child1][220] = molecule[random_child2][220]
                copy_molecules[random_child1][221] = molecule[random_child2][221]
                copy_molecules[random_child1][222] = molecule[random_child2][222]
                copy_molecules[random_child1][223] = molecule[random_child2][223]
                copy_molecules[random_child1][224] = molecule[random_child2][224]
                copy_molecules[random_child1][225] = molecule[random_child2][225]
                copy_molecules[random_child1][226] = molecule[random_child2][226]
                copy_molecules[random_child1][227] = molecule[random_child2][227]
                copy_molecules[random_child1][228] = molecule[random_child2][228]
                copy_molecules[random_child1][229] = molecule[random_child2][229]
                copy_molecules[random_child1][230] = molecule[random_child2][230]
                copy_molecules[random_child1][231] = molecule[random_child2][231]
                copy_molecules[random_child1][232] = molecule[random_child2][232]
                copy_molecules[random_child1][233] = molecule[random_child2][233]
                copy_molecules[random_child1][234] = molecule[random_child2][234]
                copy_molecules[random_child1][235] = molecule[random_child2][235]
                copy_molecules[random_child1][236] = molecule[random_child2][236]
                copy_molecules[random_child1][237] = molecule[random_child2][237]
                copy_molecules[random_child1][238] = molecule[random_child2][238]
                copy_molecules[random_child1][239] = molecule[random_child2][239]
                copy_molecules[random_child1][240] = molecule[random_child2][240]
                copy_molecules[random_child1][241] = molecule[random_child2][241]
                copy_molecules[random_child1][242] = molecule[random_child2][242]
                copy_molecules[random_child1][243] = molecule[random_child2][243]
                copy_molecules[random_child1][244] = molecule[random_child2][244]
                copy_molecules[random_child1][245] = molecule[random_child2][245]
                copy_molecules[random_child1][246] = molecule[random_child2][246]
                copy_molecules[random_child1][247] = molecule[random_child2][247]
                copy_molecules[random_child1][248] = molecule[random_child2][248]
                copy_molecules[random_child1][249] = molecule[random_child2][249]
                copy_molecules[random_child1][250] = molecule[random_child2][250]
                copy_molecules[random_child1][251] = molecule[random_child2][251]
                copy_molecules[random_child1][252] = molecule[random_child2][252]
                copy_molecules[random_child1][253] = molecule[random_child2][253]
                copy_molecules[random_child1][254] = molecule[random_child2][254]
                copy_molecules[random_child1][255] = molecule[random_child2][255]
                copy_molecules[random_child1][256] = molecule[random_child2][256]
                copy_molecules[random_child1][257] = molecule[random_child2][257]
                copy_molecules[random_child1][258] = molecule[random_child2][258]
                copy_molecules[random_child1][259] = molecule[random_child2][259]
                copy_molecules[random_child1][260] = molecule[random_child2][260]
                copy_molecules[random_child1][261] = molecule[random_child2][261]
                copy_molecules[random_child1][262] = molecule[random_child2][262]
                copy_molecules[random_child1][263] = molecule[random_child2][263]
                copy_molecules[random_child1][264] = molecule[random_child2][264]
                copy_molecules[random_child1][265] = molecule[random_child2][265]
                copy_molecules[random_child1][266] = molecule[random_child2][266]
                copy_molecules[random_child1][267] = molecule[random_child2][267]
                copy_molecules[random_child1][268] = molecule[random_child2][268]
                copy_molecules[random_child1][269] = molecule[random_child2][269]
                copy_molecules[random_child1][270] = molecule[random_child2][270]
                copy_molecules[random_child1][271] = molecule[random_child2][271]
                copy_molecules[random_child1][272] = molecule[random_child2][272]
                copy_molecules[random_child1][273] = molecule[random_child2][273]
                copy_molecules[random_child1][274] = molecule[random_child2][274]
                copy_molecules[random_child1][275] = molecule[random_child2][275]
                copy_molecules[random_child1][276] = molecule[random_child2][276]
                copy_molecules[random_child1][277] = molecule[random_child2][277]
                copy_molecules[random_child1][278] = molecule[random_child2][278]
                copy_molecules[random_child1][279] = molecule[random_child2][279]
                copy_molecules[random_child1][280] = molecule[random_child2][280]
                copy_molecules[random_child1][281] = molecule[random_child2][281]
                copy_molecules[random_child1][282] = molecule[random_child2][282]
                copy_molecules[random_child1][283] = molecule[random_child2][283]
                copy_molecules[random_child1][284] = molecule[random_child2][284]
                copy_molecules[random_child1][285] = molecule[random_child2][285]
                copy_molecules[random_child1][286] = molecule[random_child2][286]

                similar = 0
                for j in range(287):
                    if copy_molecules[random_child1][j] == 0:
                        similar = similar + 1
                if similar >= 286:
                    print('Ignore synthesis for 1 or 0 feature: ')
                    print(copy_molecules[random_child1])
                    continue
                similar = 0
                for j in range(287):
                    if copy_molecules[random_child2][j] == 0:
                        similar = similar + 1

                print('Child: {} '.format(copy_molecules[random_child1]))

                acc_result_child_1 = cal_child_accuracy(self, copy_molecules, random_child1)

                child1_obj_func_gain = objective_unction_gain_check(self,copy_molecules,random_child1, acc_result_child_1)

                acc_result_parent1 = PE[random_child1]
                acc_result_parent2 = PE[random_child2]
                KE_result_parent1 = KE[random_child1]
                KE_result_parent2 = KE[random_child2]
                if (acc_result_parent1 + acc_result_parent2 + KE_result_parent1 + KE_result_parent2) > acc_result_child_1:
                    KE[random_child1] = (acc_result_parent1 + acc_result_parent2 + KE_result_parent1 + KE_result_parent2) - acc_result_child_1
                    PE[random_child1] = acc_result_child_1
                    NumHit[random_child1] = 0
                    print('Child objective gain = {} Min objective gain = {}'.format(child1_obj_func_gain,Min_obj_func_val))
                    if (Min_obj_func_val > child1_obj_func_gain):
                        print('Synthesis Got Imptovement. Child Acc = {} MinPE = {} and P_MinPE = {}: '.format(acc_result_child_1,MinPE,P_MinPE))
                        MinPE = acc_result_child_1
                        Min_obj_func_val = child1_obj_func_gain
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
                random_position = random.randint(0, 286)
                if copy_molecules[random_child1][random_position] == 1:
                    copy_molecules[random_child1][random_position] = 0
                else:
                    copy_molecules[random_child1][random_position] = 1

                similar = 0
                for j in range(287):
                    if copy_molecules[random_child1][j] == 0:
                        similar = similar + 1
                if similar >= 286:
                    print('Ignore Intermolecule reaction for 1 or 0 features: ')
                    print(copy_molecules[random_child1])
                    continue
                similar = 0
                for j in range(287):
                    if copy_molecules[random_child2][j] == 0:
                        similar = similar + 1
                if similar >= 286:
                    print('Ignore Intermolecule reaction for 1 or 0 features: ')
                    print(copy_molecules[random_child2])
                    continue
                print('Child1: {} Child2: {}'.format(copy_molecules[random_child1], copy_molecules[random_child2]))
                acc_result_child_1 = cal_child_accuracy(self, copy_molecules, random_child1)

                child1_obj_func_gain = objective_unction_gain_check(self,copy_molecules,random_child1,acc_result_child_1)
                acc_result_child_2 = cal_child_accuracy(self, copy_molecules2, random_child2)

                child2_obj_func_gain = objective_unction_gain_check(self,copy_molecules2,random_child2,acc_result_child_2)
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
                    print('Child 1 objective gain = {} Child 2 obj gain = {} Min objective gain = {}'.format(child1_obj_func_gain,child2_obj_func_gain,Min_obj_func_val))

                    if (child1_obj_func_gain < Min_obj_func_val and child1_obj_func_gain < child2_obj_func_gain):
                        print('Inter Molecule Got Imptovement. Child 1 Acc = {} MinPE = {} and P_MinPE = {}: '.format(acc_result_child_1, MinPE, P_MinPE))
                        MinPE = acc_result_child_1
                        Min_obj_func_val = child1_obj_func_gain
                        MinStruct = molecule[random_child1].copy()
                        MinHit = NumHit[random_child1]
                        T_MinHit = T_NumHit
                    if (child2_obj_func_gain < Min_obj_func_val and child2_obj_func_gain < child1_obj_func_gain):
                        print('Inter Molecule Got Imptovement. Child 2 Acc = {} MinPE = {} and P_MinPE = {}: '.format(acc_result_child_1, MinPE, P_MinPE))
                        MinPE = acc_result_child_2
                        Min_obj_func_val = child2_obj_func_gain
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
    for j in range(287):
        if child_Mol[child][j] == 1 and j == 0:
            features.append('SL')
        elif child_Mol[child][j] == 1 and j == 1:
            features.append('DC')
        elif child_Mol[child][j] == 1 and j == 2:
            features.append('EC')
        elif child_Mol[child][j] == 1 and j == 3:
            features.append('IC')
        elif child_Mol[child][j] == 1 and j == 4:
            features.append('LAC')
        elif child_Mol[child][j] == 1 and j == 5:
            features.append('BC')
        elif child_Mol[child][j] == 1 and j == 6:
            features.append('CC')
        elif child_Mol[child][j] == 1 and j == 7:
            features.append('NC')
        elif child_Mol[child][j] == 1 and j == 8:
            features.append('ADA2')
        elif child_Mol[child][j] == 1 and j == 9:
            features.append('ADR1')
        elif child_Mol[child][j] == 1 and j == 10:
            features.append('AFT2')
        elif child_Mol[child][j] == 1 and j == 11:
            features.append('PC1')
        elif child_Mol[child][j] == 1 and j == 12:
            features.append('PC2')
        elif child_Mol[child][j] == 1 and j == 13:
            features.append('PC3')
        elif child_Mol[child][j] == 1 and j == 14:
            features.append('SL_DC_SUM')
        elif child_Mol[child][j] == 1 and j == 15:
            features.append('SL_DC_SUB')
        elif child_Mol[child][j] == 1 and j == 16:
            features.append('SL_DC_PRO')
        elif child_Mol[child][j] == 1 and j == 17:
            features.append('SL_EC_SUM')
        elif child_Mol[child][j] == 1 and j == 18:
            features.append('SL_EC_SUB')
        elif child_Mol[child][j] == 1 and j == 19:
            features.append('SL_EC_PRO')
        elif child_Mol[child][j] == 1 and j == 20:
            features.append('SL_IC_SUM')
        elif child_Mol[child][j] == 1 and j == 21:
            features.append('SL_IC_SUB')
        elif child_Mol[child][j] == 1 and j == 22:
            features.append('SL_IC_PRO')
        elif child_Mol[child][j] == 1 and j == 23:
            features.append('SL_LAC_SUM')
        elif child_Mol[child][j] == 1 and j == 24:
            features.append('SL_LAC_SUB')
        elif child_Mol[child][j] == 1 and j == 25:
            features.append('SL_LAC_PRO')
        elif child_Mol[child][j] == 1 and j == 26:
            features.append('SL_BC_SUM')
        elif child_Mol[child][j] == 1 and j == 27:
            features.append('SL_BC_SUB')
        elif child_Mol[child][j] == 1 and j == 28:
            features.append('SL_BC_PRO')
        elif child_Mol[child][j] == 1 and j == 29:
            features.append('SL_CC_SUM')
        elif child_Mol[child][j] == 1 and j == 30:
            features.append('SL_CC_SUB')
        elif child_Mol[child][j] == 1 and j == 31:
            features.append('SL_CC_PRO')
        elif child_Mol[child][j] == 1 and j == 32:
            features.append('SL_NC_SUM')
        elif child_Mol[child][j] == 1 and j == 33:
            features.append('SL_NC_SUB')
        elif child_Mol[child][j] == 1 and j == 34:
            features.append('SL_NC_PRO')
        elif child_Mol[child][j] == 1 and j == 35:
            features.append('SL_ADA2_SUM')
        elif child_Mol[child][j] == 1 and j == 36:
            features.append('SL_ADA2_SUB')
        elif child_Mol[child][j] == 1 and j == 37:
            features.append('SL_ADA2_PRO')
        elif child_Mol[child][j] == 1 and j == 38:
            features.append('SL_ADR1_SUM')
        elif child_Mol[child][j] == 1 and j == 39:
            features.append('SL_ADR1_SUB')
        elif child_Mol[child][j] == 1 and j == 40:
            features.append('SL_ADR1_PRO')
        elif child_Mol[child][j] == 1 and j == 41:
            features.append('SL_AFT2_SUM')
        elif child_Mol[child][j] == 1 and j == 42:
            features.append('SL_AFT2_SUB')
        elif child_Mol[child][j] == 1 and j == 43:
            features.append('SL_AFT2_PRO')
        elif child_Mol[child][j] == 1 and j == 44:
            features.append('SL_PC1_SUM')
        elif child_Mol[child][j] == 1 and j == 45:
            features.append('SL_PC1_SUB')
        elif child_Mol[child][j] == 1 and j == 46:
            features.append('SL_PC1_PRO')
        elif child_Mol[child][j] == 1 and j == 47:
            features.append('SL_PC2_SUM')
        elif child_Mol[child][j] == 1 and j == 48:
            features.append('SL_PC2_SUB')
        elif child_Mol[child][j] == 1 and j == 49:
            features.append('SL_PC2_PRO')
        elif child_Mol[child][j] == 1 and j == 50:
            features.append('SL_PC3_SUM')
        elif child_Mol[child][j] == 1 and j == 51:
            features.append('SL_PC3_SUB')
        elif child_Mol[child][j] == 1 and j == 52:
            features.append('SL_PC3_PRO')
        elif child_Mol[child][j] == 1 and j == 53:
            features.append('DC_EC_SUM')
        elif child_Mol[child][j] == 1 and j == 54:
            features.append('DC_EC_SUB')
        elif child_Mol[child][j] == 1 and j == 55:
            features.append('DC_EC_PRO')
        elif child_Mol[child][j] == 1 and j == 56:
            features.append('DC_IC_SUM')
        elif child_Mol[child][j] == 1 and j == 57:
            features.append('DC_IC_SUB')
        elif child_Mol[child][j] == 1 and j == 58:
            features.append('DC_IC_PRO')
        elif child_Mol[child][j] == 1 and j == 59:
            features.append('DC_LAC_SUM')
        elif child_Mol[child][j] == 1 and j == 60:
            features.append('DC_LAC_SUB')
        elif child_Mol[child][j] == 1 and j == 61:
            features.append('DC_LAC_PRO')
        elif child_Mol[child][j] == 1 and j == 62:
            features.append('DC_BC_SUM')
        elif child_Mol[child][j] == 1 and j == 63:
            features.append('DC_BC_SUB')
        elif child_Mol[child][j] == 1 and j == 64:
            features.append('DC_BC_PRO')
        elif child_Mol[child][j] == 1 and j == 65:
            features.append('DC_CC_SUM')
        elif child_Mol[child][j] == 1 and j == 66:
            features.append('DC_CC_SUB')
        elif child_Mol[child][j] == 1 and j == 67:
            features.append('DC_CC_PRO')
        elif child_Mol[child][j] == 1 and j == 68:
            features.append('DC_NC_SUM')
        elif child_Mol[child][j] == 1 and j == 69:
            features.append('DC_NC_SUB')
        elif child_Mol[child][j] == 1 and j == 70:
            features.append('DC_NC_PRO')
        elif child_Mol[child][j] == 1 and j == 71:
            features.append('DC_ADA2_SUM')
        elif child_Mol[child][j] == 1 and j == 72:
            features.append('DC_ADA2_SUB')
        elif child_Mol[child][j] == 1 and j == 73:
            features.append('DC_ADA2_PRO')
        elif child_Mol[child][j] == 1 and j == 74:
            features.append('DC_ADR1_SUM')
        elif child_Mol[child][j] == 1 and j == 75:
            features.append('DC_ADR1_SUB')
        elif child_Mol[child][j] == 1 and j == 76:
            features.append('DC_ADR1_PRO')
        elif child_Mol[child][j] == 1 and j == 77:
            features.append('DC_AFT2_SUM')
        elif child_Mol[child][j] == 1 and j == 78:
            features.append('DC_AFT2_SUB')
        elif child_Mol[child][j] == 1 and j == 79:
            features.append('DC_AFT2_PRO')
        elif child_Mol[child][j] == 1 and j == 80:
            features.append('DC_PC1_SUM')
        elif child_Mol[child][j] == 1 and j == 81:
            features.append('DC_PC1_SUB')
        elif child_Mol[child][j] == 1 and j == 82:
            features.append('DC_PC1_PRO')
        elif child_Mol[child][j] == 1 and j == 83:
            features.append('DC_PC2_SUM')
        elif child_Mol[child][j] == 1 and j == 84:
            features.append('DC_PC2_SUB')
        elif child_Mol[child][j] == 1 and j == 85:
            features.append('DC_PC2_PRO')
        elif child_Mol[child][j] == 1 and j == 86:
            features.append('DC_PC3_SUM')
        elif child_Mol[child][j] == 1 and j == 87:
            features.append('DC_PC3_SUB')
        elif child_Mol[child][j] == 1 and j == 88:
            features.append('DC_PC3_PRO')
        elif child_Mol[child][j] == 1 and j == 89:
            features.append('EC_IC_SUM')
        elif child_Mol[child][j] == 1 and j == 90:
            features.append('EC_IC_SUB')
        elif child_Mol[child][j] == 1 and j == 91:
            features.append('EC_IC_PRO')
        elif child_Mol[child][j] == 1 and j == 92:
            features.append('EC_LAC_SUM')
        elif child_Mol[child][j] == 1 and j == 93:
            features.append('EC_LAC_SUB')
        elif child_Mol[child][j] == 1 and j == 94:
            features.append('EC_LAC_PRO')
        elif child_Mol[child][j] == 1 and j == 95:
            features.append('EC_BC_SUM')
        elif child_Mol[child][j] == 1 and j == 96:
            features.append('EC_BC_SUB')
        elif child_Mol[child][j] == 1 and j == 97:
            features.append('EC_BC_PRO')
        elif child_Mol[child][j] == 1 and j == 98:
            features.append('EC_CC_SUM')
        elif child_Mol[child][j] == 1 and j == 99:
            features.append('EC_CC_SUB')
        elif child_Mol[child][j] == 1 and j == 100:
            features.append('EC_CC_PRO')
        elif child_Mol[child][j] == 1 and j == 101:
            features.append('EC_NC_SUM')
        elif child_Mol[child][j] == 1 and j == 102:
            features.append('EC_NC_SUB')
        elif child_Mol[child][j] == 1 and j == 103:
            features.append('EC_NC_PRO')
        elif child_Mol[child][j] == 1 and j == 104:
            features.append('EC_ADA2_SUM')
        elif child_Mol[child][j] == 1 and j == 105:
            features.append('EC_ADA2_SUB')
        elif child_Mol[child][j] == 1 and j == 106:
            features.append('EC_ADA2_PRO')
        elif child_Mol[child][j] == 1 and j == 107:
            features.append('EC_ADR1_SUM')
        elif child_Mol[child][j] == 1 and j == 108:
            features.append('EC_ADR1_SUB')
        elif child_Mol[child][j] == 1 and j == 109:
            features.append('EC_ADR1_PRO')
        elif child_Mol[child][j] == 1 and j == 110:
            features.append('EC_AFT2_SUM')
        elif child_Mol[child][j] == 1 and j == 111:
            features.append('EC_AFT2_SUB')
        elif child_Mol[child][j] == 1 and j == 112:
            features.append('EC_AFT2_PRO')
        elif child_Mol[child][j] == 1 and j == 113:
            features.append('EC_PC1_SUM')
        elif child_Mol[child][j] == 1 and j == 114:
            features.append('EC_PC1_SUB')
        elif child_Mol[child][j] == 1 and j == 115:
            features.append('EC_PC1_PRO')
        elif child_Mol[child][j] == 1 and j == 116:
            features.append('EC_PC2_SUM')
        elif child_Mol[child][j] == 1 and j == 117:
            features.append('EC_PC2_SUB')
        elif child_Mol[child][j] == 1 and j == 118:
            features.append('EC_PC2_PRO')
        elif child_Mol[child][j] == 1 and j == 119:
            features.append('EC_PC3_SUM')
        elif child_Mol[child][j] == 1 and j == 120:
            features.append('EC_PC3_SUB')
        elif child_Mol[child][j] == 1 and j == 121:
            features.append('EC_PC3_PRO')
        elif child_Mol[child][j] == 1 and j == 122:
            features.append('IC_LAC_SUM')
        elif child_Mol[child][j] == 1 and j == 123:
            features.append('IC_LAC_SUB')
        elif child_Mol[child][j] == 1 and j == 124:
            features.append('IC_LAC_PRO')
        elif child_Mol[child][j] == 1 and j == 125:
            features.append('IC_BC_SUM')
        elif child_Mol[child][j] == 1 and j == 126:
            features.append('IC_BC_SUB')
        elif child_Mol[child][j] == 1 and j == 127:
            features.append('IC_BC_PRO')
        elif child_Mol[child][j] == 1 and j == 128:
            features.append('IC_CC_SUM')
        elif child_Mol[child][j] == 1 and j == 129:
            features.append('IC_CC_SUB')
        elif child_Mol[child][j] == 1 and j == 130:
            features.append('IC_CC_PRO')
        elif child_Mol[child][j] == 1 and j == 131:
            features.append('IC_NC_SUM')
        elif child_Mol[child][j] == 1 and j == 132:
            features.append('IC_NC_SUB')
        elif child_Mol[child][j] == 1 and j == 133:
            features.append('IC_NC_PRO')
        elif child_Mol[child][j] == 1 and j == 134:
            features.append('IC_ADA2_SUM')
        elif child_Mol[child][j] == 1 and j == 135:
            features.append('IC_ADA2_SUB')
        elif child_Mol[child][j] == 1 and j == 136:
            features.append('IC_ADA2_PRO')
        elif child_Mol[child][j] == 1 and j == 137:
            features.append('IC_ADR1_SUM')
        elif child_Mol[child][j] == 1 and j == 138:
            features.append('IC_ADR1_SUB')
        elif child_Mol[child][j] == 1 and j == 139:
            features.append('IC_ADR1_PRO')
        elif child_Mol[child][j] == 1 and j == 140:
            features.append('IC_AFT2_SUM')
        elif child_Mol[child][j] == 1 and j == 141:
            features.append('IC_AFT2_SUB')
        elif child_Mol[child][j] == 1 and j == 142:
            features.append('IC_AFT2_PRO')
        elif child_Mol[child][j] == 1 and j == 143:
            features.append('IC_PC1_SUM')
        elif child_Mol[child][j] == 1 and j == 144:
            features.append('IC_PC1_SUB')
        elif child_Mol[child][j] == 1 and j == 145:
            features.append('IC_PC1_PRO')
        elif child_Mol[child][j] == 1 and j == 146:
            features.append('IC_PC2_SUM')
        elif child_Mol[child][j] == 1 and j == 147:
            features.append('IC_PC2_SUB')
        elif child_Mol[child][j] == 1 and j == 148:
            features.append('IC_PC2_PRO')
        elif child_Mol[child][j] == 1 and j == 149:
            features.append('IC_PC3_SUM')
        elif child_Mol[child][j] == 1 and j == 150:
            features.append('IC_PC3_SUB')
        elif child_Mol[child][j] == 1 and j == 151:
            features.append('IC_PC3_PRO')
        elif child_Mol[child][j] == 1 and j == 152:
            features.append('LAC_BC_SUM')
        elif child_Mol[child][j] == 1 and j == 153:
            features.append('LAC_BC_SUB')
        elif child_Mol[child][j] == 1 and j == 154:
            features.append('LAC_BC_PRO')
        elif child_Mol[child][j] == 1 and j == 155:
            features.append('LAC_CC_SUM')
        elif child_Mol[child][j] == 1 and j == 156:
            features.append('LAC_CC_SUB')
        elif child_Mol[child][j] == 1 and j == 157:
            features.append('LAC_CC_PRO')
        elif child_Mol[child][j] == 1 and j == 158:
            features.append('LAC_NC_SUM')
        elif child_Mol[child][j] == 1 and j == 159:
            features.append('LAC_NC_SUB')
        elif child_Mol[child][j] == 1 and j == 160:
            features.append('LAC_NC_PRO')
        elif child_Mol[child][j] == 1 and j == 161:
            features.append('LAC_ADA2_SUM')
        elif child_Mol[child][j] == 1 and j == 162:
            features.append('LAC_ADA2_SUB')
        elif child_Mol[child][j] == 1 and j == 163:
            features.append('LAC_ADA2_PRO')
        elif child_Mol[child][j] == 1 and j == 164:
            features.append('LAC_ADR1_SUM')
        elif child_Mol[child][j] == 1 and j == 165:
            features.append('LAC_ADR1_SUB')
        elif child_Mol[child][j] == 1 and j == 166:
            features.append('LAC_ADR1_PRO')
        elif child_Mol[child][j] == 1 and j == 167:
            features.append('LAC_AFT2_SUM')
        elif child_Mol[child][j] == 1 and j == 168:
            features.append('LAC_AFT2_SUB')
        elif child_Mol[child][j] == 1 and j == 169:
            features.append('LAC_AFT2_PRO')
        elif child_Mol[child][j] == 1 and j == 170:
            features.append('LAC_PC1_SUM')
        elif child_Mol[child][j] == 1 and j == 171:
            features.append('LAC_PC1_SUB')
        elif child_Mol[child][j] == 1 and j == 172:
            features.append('LAC_PC1_PRO')
        elif child_Mol[child][j] == 1 and j == 173:
            features.append('LAC_PC2_SUM')
        elif child_Mol[child][j] == 1 and j == 174:
            features.append('LAC_PC2_SUB')
        elif child_Mol[child][j] == 1 and j == 175:
            features.append('LAC_PC2_PRO')
        elif child_Mol[child][j] == 1 and j == 176:
            features.append('LAC_PC3_SUM')
        elif child_Mol[child][j] == 1 and j == 177:
            features.append('LAC_PC3_SUB')
        elif child_Mol[child][j] == 1 and j == 178:
            features.append('LAC_PC3_PRO')
        elif child_Mol[child][j] == 1 and j == 179:
            features.append('BC_CC_SUM')
        elif child_Mol[child][j] == 1 and j == 180:
            features.append('BC_CC_SUB')
        elif child_Mol[child][j] == 1 and j == 181:
            features.append('BC_CC_PRO')
        elif child_Mol[child][j] == 1 and j == 182:
            features.append('BC_NC_SUM')
        elif child_Mol[child][j] == 1 and j == 183:
            features.append('BC_NC_SUB')
        elif child_Mol[child][j] == 1 and j == 184:
            features.append('BC_NC_PRO')
        elif child_Mol[child][j] == 1 and j == 185:
            features.append('BC_ADA2_SUM')
        elif child_Mol[child][j] == 1 and j == 186:
            features.append('BC_ADA2_SUB')
        elif child_Mol[child][j] == 1 and j == 187:
            features.append('BC_ADA2_PRO')
        elif child_Mol[child][j] == 1 and j == 188:
            features.append('BC_ADR1_SUM')
        elif child_Mol[child][j] == 1 and j == 189:
            features.append('BC_ADR1_SUB')
        elif child_Mol[child][j] == 1 and j == 190:
            features.append('BC_ADR1_PRO')
        elif child_Mol[child][j] == 1 and j == 191:
            features.append('BC_AFT2_SUM')
        elif child_Mol[child][j] == 1 and j == 192:
            features.append('BC_AFT2_SUB')
        elif child_Mol[child][j] == 1 and j == 193:
            features.append('BC_AFT2_PRO')
        elif child_Mol[child][j] == 1 and j == 194:
            features.append('BC_PC1_SUM')
        elif child_Mol[child][j] == 1 and j == 195:
            features.append('BC_PC1_SUB')
        elif child_Mol[child][j] == 1 and j == 196:
            features.append('BC_PC1_PRO')
        elif child_Mol[child][j] == 1 and j == 197:
            features.append('BC_PC2_SUM')
        elif child_Mol[child][j] == 1 and j == 198:
            features.append('BC_PC2_SUB')
        elif child_Mol[child][j] == 1 and j == 199:
            features.append('BC_PC2_PRO')
        elif child_Mol[child][j] == 1 and j == 200:
            features.append('BC_PC3_SUM')
        elif child_Mol[child][j] == 1 and j == 201:
            features.append('BC_PC3_SUB')
        elif child_Mol[child][j] == 1 and j == 202:
            features.append('BC_PC3_PRO')
        elif child_Mol[child][j] == 1 and j == 203:
            features.append('CC_NC_SUM')
        elif child_Mol[child][j] == 1 and j == 204:
            features.append('CC_NC_SUB')
        elif child_Mol[child][j] == 1 and j == 205:
            features.append('CC_NC_PRO')
        elif child_Mol[child][j] == 1 and j == 206:
            features.append('CC_ADA2_SUM')
        elif child_Mol[child][j] == 1 and j == 207:
            features.append('CC_ADA2_SUB')
        elif child_Mol[child][j] == 1 and j == 208:
            features.append('CC_ADA2_PRO')
        elif child_Mol[child][j] == 1 and j == 209:
            features.append('CC_ADR1_SUM')
        elif child_Mol[child][j] == 1 and j == 210:
            features.append('CC_ADR1_SUB')
        elif child_Mol[child][j] == 1 and j == 211:
            features.append('CC_ADR1_PRO')
        elif child_Mol[child][j] == 1 and j == 212:
            features.append('CC_AFT2_SUM')
        elif child_Mol[child][j] == 1 and j == 213:
            features.append('CC_AFT2_SUB')
        elif child_Mol[child][j] == 1 and j == 214:
            features.append('CC_AFT2_PRO')
        elif child_Mol[child][j] == 1 and j == 215:
            features.append('CC_PC1_SUM')
        elif child_Mol[child][j] == 1 and j == 216:
            features.append('CC_PC1_SUB')
        elif child_Mol[child][j] == 1 and j == 217:
            features.append('CC_PC1_PRO')
        elif child_Mol[child][j] == 1 and j == 218:
            features.append('CC_PC2_SUM')
        elif child_Mol[child][j] == 1 and j == 219:
            features.append('CC_PC2_SUB')
        elif child_Mol[child][j] == 1 and j == 220:
            features.append('CC_PC2_PRO')
        elif child_Mol[child][j] == 1 and j == 221:
            features.append('CC_PC3_SUM')
        elif child_Mol[child][j] == 1 and j == 222:
            features.append('CC_PC3_SUB')
        elif child_Mol[child][j] == 1 and j == 223:
            features.append('CC_PC3_PRO')
        elif child_Mol[child][j] == 1 and j == 224:
            features.append('NC_ADA2_SUM')
        elif child_Mol[child][j] == 1 and j == 225:
            features.append('NC_ADA2_SUB')
        elif child_Mol[child][j] == 1 and j == 226:
            features.append('NC_ADA2_PRO')
        elif child_Mol[child][j] == 1 and j == 227:
            features.append('NC_ADR1_SUM')
        elif child_Mol[child][j] == 1 and j == 228:
            features.append('NC_ADR1_SUB')
        elif child_Mol[child][j] == 1 and j == 229:
            features.append('NC_ADR1_PRO')
        elif child_Mol[child][j] == 1 and j == 230:
            features.append('NC_AFT2_SUM')
        elif child_Mol[child][j] == 1 and j == 231:
            features.append('NC_AFT2_SUB')
        elif child_Mol[child][j] == 1 and j == 232:
            features.append('NC_AFT2_PRO')
        elif child_Mol[child][j] == 1 and j == 233:
            features.append('NC_PC1_SUM')
        elif child_Mol[child][j] == 1 and j == 234:
            features.append('NC_PC1_SUB')
        elif child_Mol[child][j] == 1 and j == 235:
            features.append('NC_PC1_PRO')
        elif child_Mol[child][j] == 1 and j == 236:
            features.append('NC_PC2_SUM')
        elif child_Mol[child][j] == 1 and j == 237:
            features.append('NC_PC2_SUB')
        elif child_Mol[child][j] == 1 and j == 238:
            features.append('NC_PC2_PRO')
        elif child_Mol[child][j] == 1 and j == 239:
            features.append('NC_PC3_SUM')
        elif child_Mol[child][j] == 1 and j == 240:
            features.append('NC_PC3_SUB')
        elif child_Mol[child][j] == 1 and j == 241:
            features.append('NC_PC3_PRO')
        elif child_Mol[child][j] == 1 and j == 242:
            features.append('ADA2_ADR1_SUM')
        elif child_Mol[child][j] == 1 and j == 243:
            features.append('ADA2_ADR1_SUB')
        elif child_Mol[child][j] == 1 and j == 244:
            features.append('ADA2_ADR1_PRO')
        elif child_Mol[child][j] == 1 and j == 245:
            features.append('ADA2_AFT2_SUM')
        elif child_Mol[child][j] == 1 and j == 246:
            features.append('ADA2_AFT2_SUB')
        elif child_Mol[child][j] == 1 and j == 247:
            features.append('ADA2_AFT2_PRO')
        elif child_Mol[child][j] == 1 and j == 248:
            features.append('ADA2_PC1_SUM')
        elif child_Mol[child][j] == 1 and j == 249:
            features.append('ADA2_PC1_SUB')
        elif child_Mol[child][j] == 1 and j == 250:
            features.append('ADA2_PC1_PRO')
        elif child_Mol[child][j] == 1 and j == 251:
            features.append('ADA2_PC2_SUM')
        elif child_Mol[child][j] == 1 and j == 252:
            features.append('ADA2_PC2_SUB')
        elif child_Mol[child][j] == 1 and j == 253:
            features.append('ADA2_PC2_PRO')
        elif child_Mol[child][j] == 1 and j == 254:
            features.append('ADA2_PC3_SUM')
        elif child_Mol[child][j] == 1 and j == 255:
            features.append('ADA2_PC3_SUB')
        elif child_Mol[child][j] == 1 and j == 256:
            features.append('ADA2_PC3_PRO')
        elif child_Mol[child][j] == 1 and j == 257:
            features.append('ADR1_AFT2_SUM')
        elif child_Mol[child][j] == 1 and j == 258:
            features.append('ADR1_AFT2_SUB')
        elif child_Mol[child][j] == 1 and j == 259:
            features.append('ADR1_AFT2_PRO')
        elif child_Mol[child][j] == 1 and j == 260:
            features.append('ADR1_PC1_SUM')
        elif child_Mol[child][j] == 1 and j == 261:
            features.append('ADR1_PC1_SUB')
        elif child_Mol[child][j] == 1 and j == 262:
            features.append('ADR1_PC1_PRO')
        elif child_Mol[child][j] == 1 and j == 263:
            features.append('ADR1_PC2_SUM')
        elif child_Mol[child][j] == 1 and j == 264:
            features.append('ADR1_PC2_SUB')
        elif child_Mol[child][j] == 1 and j == 265:
            features.append('ADR1_PC2_PRO')
        elif child_Mol[child][j] == 1 and j == 266:
            features.append('ADR1_PC3_SUM')
        elif child_Mol[child][j] == 1 and j == 267:
            features.append('ADR1_PC3_SUB')
        elif child_Mol[child][j] == 1 and j == 268:
            features.append('ADR1_PC3_PRO')
        elif child_Mol[child][j] == 1 and j == 269:
            features.append('AFT2_PC1_SUM')
        elif child_Mol[child][j] == 1 and j == 270:
            features.append('AFT2_PC1_SUB')
        elif child_Mol[child][j] == 1 and j == 271:
            features.append('AFT2_PC1_PRO')
        elif child_Mol[child][j] == 1 and j == 272:
            features.append('AFT2_PC2_SUM')
        elif child_Mol[child][j] == 1 and j == 273:
            features.append('AFT2_PC2_SUB')
        elif child_Mol[child][j] == 1 and j == 274:
            features.append('AFT2_PC2_PRO')
        elif child_Mol[child][j] == 1 and j == 275:
            features.append('AFT2_PC3_SUM')
        elif child_Mol[child][j] == 1 and j == 276:
            features.append('AFT2_PC3_SUB')
        elif child_Mol[child][j] == 1 and j == 277:
            features.append('AFT2_PC3_PRO')
        elif child_Mol[child][j] == 1 and j == 278:
            features.append('PC1_PC2_SUM')
        elif child_Mol[child][j] == 1 and j == 279:
            features.append('PC1_PC2_SUB')
        elif child_Mol[child][j] == 1 and j == 280:
            features.append('PC1_PC2_PRO')
        elif child_Mol[child][j] == 1 and j == 281:
            features.append('PC1_PC3_SUM')
        elif child_Mol[child][j] == 1 and j == 282:
            features.append('PC1_PC3_SUB')
        elif child_Mol[child][j] == 1 and j == 283:
            features.append('PC1_PC3_PRO')
        elif child_Mol[child][j] == 1 and j == 284:
            features.append('PC2_PC3_SUM')
        elif child_Mol[child][j] == 1 and j == 285:
            features.append('PC2_PC3_SUB')
        elif child_Mol[child][j] == 1 and j == 286:
            features.append('PC2_PC3_PRO')

    X = df[features]
    y = df['P_Status']
    # Remove comments from 2924 and 2925 lines to execute the program using the imbalanced dataset.
    # Make comments to 2929 and 2930 lines.
    #X_smaote = X
    #y_smote = y
    # Remove comments from 2929 and 2930 lines.
    # Make comments to line 2925 and 2926 to execute the program using the balanced dataset.
    smote = SMOTEENN(sampling_strategy='minority')
    X_smote, y_smote = smote.fit_resample(X, y)

    # Remove comment from line 2934 to execute the program using LightGBM Classifier.
    # Remove comment from line 2944 to line 2946. Make comments from line 2950 to line 2964.
    #clf = lgb.LGBMClassifier()
    # Remove comment from line 2937 to execute the program using XGBoost Classifier.
    # Remove comment from line 2944 to line 2946. Make comments from line 2950 to line 2964.
    #clf = XGBClassifier()
    # Remove comment from line 2940 to execute the program using Random Forest Classifier.
    # Remove comment from line 2944 to line 2946. Make comments from line 2950 to line 2964.
    clf = RandomForestClassifier(n_estimators=100)

    # Remove comment from line 2934 to line 2936 to execute the program  either using LightGBM, XGBoost Classifier, or Random Forest Classifier.
    # Make comments from line 2950 to line 2964.
    scores = cross_val_score(estimator=clf, X=X_smote, y=y_smote, cv=10, scoring='accuracy')
    predicted_label = cross_val_predict(estimator=clf, X=X_smote, y=y_smote, cv=10)
    score = round(scores.mean(), 6)

    # Remove comment from line 2950 to line 2964 to execute the program using ensemble method.
    # Make comments from line 2934 to line 2946.
    #estimators = []
    #estimators.clear()
    #clf = XGBClassifier(max_depth=4, scale_pos_weight=3)
    #estimators.append(('XGBoost', clf))

    #clf2 = RandomForestClassifier(n_estimators=100)
    #estimators.append(('RandomForest', clf2))

    #clf3 = lgb.LGBMClassifier()
    #estimators.append(('DecisionTree', clf3))

    #ensemble = VotingClassifier(estimators)
    #scores = cross_val_score(estimator=ensemble, X=X_smote, y=y_smote, cv=10, scoring='accuracy')
    #predicted_label = cross_val_predict(estimator=ensemble, X=X_smote, y=y_smote, cv=10)
    #score = round(scores.mean(), 6)

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
