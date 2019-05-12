#Importing libraries
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def PCATest(df, n_scor, rango, layout=(1,1), Figsize=(15,13)):

    fig, ax = plt.subplots(layout[0], layout[1], figsize=Figsize)
    
    for i, m in enumerate(n_scor):

        df = df[0:rango, :]

        interia = []

        for center in range(1,m+1):
    
            kmeans = KMeans(center, random_state=0)
    
            model = kmeans.fit(df)

            interia.append(model.inertia_)
    
        centers = list(range(1,m+1))

        plt.subplot(layout[0], layout[1], i+1) 
        plt.plot(centers,interia)
        plt.title('Kmeans (PCA components)')
        plt.xlabel('Centers');
        plt.ylabel('Average distance');
        plt.grid()
        i+=1
        
    plt.subplots_adjust(top=1, bottom=0.08, left=0.10, right=0.95, hspace=0.3, wspace=0.35);
    

def PCATestB(df, n_comp, n_scor, rango, layout=(1,1), Figsize=(15,13)):

    fig, ax = plt.subplots(layout[0], layout[1], figsize=Figsize)
    pca = PCA()
    i=1
    
    for n, m in zip(n_comp, n_scor):
    
        pca = PCA(n)
        pca.fit(df)
        A =pca.transform(df)

        A = A[0:rango, :]

        scores = []
        interia = []


        for center in range(1,m):
    
            kmeans = KMeans(center, random_state=0)
    
            model = kmeans.fit(A)
    
            scores.append(model.score(A))
            interia.append(model.inertia_)
    
        centers = list(range(1,m))

        plt.subplot(layout[0], layout[1], i) 
        plt.plot(centers,interia)
        plt.title('Kmeans (' + str(n) + ' PCA components)')
        plt.xlabel('Centers');
        plt.ylabel('Average distance');
        plt.grid()
        i+=1
        
    plt.subplots_adjust(top=1, bottom=0.08, left=0.10, right=0.95, hspace=0.3, wspace=0.35);
    
    
def PCAvariance(VarRat, VarRatAcum):
    
    fig, ax1 = plt.subplots(figsize=(15,5))

    plt.rcParams.update({'font.size': 14})

    ax1.bar(range(len(VarRat)),VarRat, color='c')
    ax1.set_xlabel('PCA components')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Variance ratio', color='k')
    ax1.tick_params('y', colors='k')
    plt.grid(axis='x')

    ax2 = ax1.twinx()
    ax2.plot(VarRatAcum, 'r-*')
    ax2.set_ylabel('Commulative Variance ratio', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    plt.grid(axis='y')
    plt.show()


def Subsets(df1, df2):
    
    fig, ax = plt.subplots(2, 1, figsize=(18,7))

    ax1 = plt.subplot(2,1,1) 
    ax1 = sns.barplot(df1.columns,df1.isnull().sum().values);
    ax1.set(xticklabels=[])
    plt.ylabel('NaN values (subset A)');
    plt.grid()

    ax2 = plt.subplot(2,1,2)
    ax2 = sns.barplot(df2.columns,df2.isnull().sum().values);
    plt.xticks(rotation=90);
    plt.ylabel('NaN values (subset B)');
    plt.grid()

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.04, wspace=0.35)

def NaNValuesDistribution(df, null_columns):
    
    fig, ax = plt.subplots(2, 1, figsize=(18,9))
    ax1 = plt.subplot(2,1,1) 
    sns.barplot(df.columns,df.isnull().sum().values);
    plt.xticks(rotation=90);
    plt.ylabel('NaN values');
    plt.grid()
    ax2 = plt.subplot(2,1,2)
    ax2 = df[null_columns].isnull().sum().hist(bins=50);
    plt.ylabel('Columns')
    plt.xlabel('Amount of NaN')

    plt.subplots_adjust(top=1, bottom=0.08, left=0.10, right=0.95, hspace=0.7, wspace=0.35);
    
def ColDistsHist(df1, df2, cols, bins):
    
    fig, ax = plt.subplots(5, 2, figsize=(12,15))

    ax1 = plt.subplot(5,2,1) 
    ax1 = plt.hist(df1[cols[0]], color='c', bins=bins[0])
    plt.title('Subset A ('+ cols[0] + ')')
    plt.xlabel('Values');
    plt.ylabel('Amount');
    plt.grid()
    plt.xticks(rotation=90);
    ax2 = plt.subplot(5,2,2) 
    ax2 = plt.hist(df2[cols[0]], color='g', bins=bins[0])
    plt.title('Subset B ('+ cols[0] + ')')
    plt.xlabel('Values');
    plt.ylabel('Amount');
    plt.grid()
    plt.xticks(rotation=90);
    ax3 = plt.subplot(5,2,3) 
    ax3 = plt.hist(df1[cols[1]], color='c', bins=bins[1])
    plt.title('Subset A ('+ cols[1] + ')')
    plt.xlabel('Values');
    plt.ylabel('Amount');
    plt.grid()
    plt.xticks(rotation=90);
    ax4 = plt.subplot(5,2,4) 
    ax4 = plt.hist(df2[cols[1]], color='g', bins=bins[1])
    plt.title('Subset B ('+ cols[1] + ')')
    plt.xlabel('Values');
    plt.ylabel('Amount');
    plt.grid()
    plt.xticks(rotation=90);
    ax5 = plt.subplot(5,2,5) 
    ax5 = plt.hist(df1[cols[2]], color='c', bins=bins[2])
    plt.title('Subset A ('+ cols[2] + ')')
    plt.xlabel('Values');
    plt.ylabel('Amount');
    plt.grid()
    plt.xticks(rotation=90);
    ax6 = plt.subplot(5,2,6) 
    ax6 = plt.hist(df2[cols[2]], color='g', bins=bins[2])
    plt.title('Subset B ('+ cols[2] + ')')
    plt.xlabel('Values');
    plt.ylabel('Amount');
    plt.grid()
    plt.xticks(rotation=90);
    ax7 = plt.subplot(5,2,7) 
    ax7 = plt.hist(df1[cols[3]], color='c', bins=bins[3])
    plt.title('Subset A ('+ cols[3] + ')')
    plt.xlabel('Values');
    plt.ylabel('Amount');
    plt.grid()
    plt.xticks(rotation=90);
    ax8 = plt.subplot(5,2,8) 
    ax8 = plt.hist(df2[cols[3]], color='g', bins=bins[3])
    plt.title('Subset B ('+ cols[3] + ')')
    plt.xlabel('Values');
    plt.ylabel('Amount');
    plt.grid()
    plt.xticks(rotation=90);
    ax9 = plt.subplot(5,2,9) 
    ax9 = plt.hist(df1[cols[4]], color='c', bins=bins[4])
    plt.title('Subset A ('+ cols[4] + ')')
    plt.xlabel('Values');
    plt.ylabel('Amount');
    plt.grid()
    plt.xticks(rotation=90);
    ax10 = plt.subplot(5,2,10) 
    ax10 = plt.hist(df2[cols[4]], color='g', bins=bins[4])    
    plt.title('Subset B ('+ cols[4] + ')')
    plt.xlabel('Values');
    plt.ylabel('Amount');
    plt.grid()    
    plt.xticks(rotation=90);
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.4, wspace=0.3)

    plt.show()
    
    
def clean_data(df, feat_df, DictAttr):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
    
    # Put in code here to execute all main cleaning steps:
    #*******************************************************************************************************************
    # convert missing value codes into NaNs, ...
    #Removing brackets in the missing values column for their processing
    feat_df.missing_or_unknown = feat_df.missing_or_unknown.str.replace('[','').str.replace(']','')

    #Removing empty missing values codes
    feat_df.drop(feat_df[feat_df['missing_or_unknown'] == ''].index, inplace=True)

    #Creating a dictionary for processing the missing values codes
    Dict = {}

    for i, attribute in zip(feat_df.index,feat_df.attribute):
        Dict[attribute] = feat_df.missing_or_unknown.loc[i].split(',')

    #Finding columns with natural missing values in the df dataframe
    null_columns=df.columns[df.isnull().any()]

    #Finding the amount of natural missing values and the missing values related with the missing codes
    NatNaN = df.isnull().sum().values
    CodNaN = np.zeros(shape=df.shape[1])

    #Changing missing values codes with nan values
    for key in list(Dict.keys()):

        for n in Dict[key]: 
            if df[key].dtype == 'O':
                suma = df[key][df[key] == n].value_counts().sum()
                CodNaN[feat_df[feat_df['attribute'] ==key].index] += suma
                df[key] = df[key].replace(n,np.nan)
            else:
                suma = df[key][df[key] == int(n)].value_counts().sum()
                CodNaN[feat_df[feat_df['attribute'] ==key].index] += suma
                df[key] = df[key].replace(int(n),np.nan)

    #*******************************************************************************************************************
    # remove selected columns and rows, ...
    
    # Perform an assessment of how much missing data there is in each column of the dataset.
    null_columns=df.columns[df.isnull().any()]
    
    # Remove the outlier columns from the dataset. (You'll perform other data
    # engineering tasks such as re-encoding and imputation later.)
    A = df.isnull().sum()
    q1 = A.quantile(0.25)
    q3 = A.quantile(0.75)
    IQran = q3-q1
    In_fence  = q1-1.5*IQran
    Out_fence = q3+1.5*IQran
    columns = df[null_columns].columns[(df[null_columns].isnull().sum() < Out_fence)]
    cols = [col for col in columns]
    df = df[cols]    
    
    # Write code to divide the data into two subsets based on the number of missing
    # values in each row.
    NullRowsA = df[(df.isnull().sum(1) <= 20)].index

    df = df.loc[NullRowsA]
   
    #*******************************************************************************************************************
    # select, re-encode, and engineer column values.
    
    multi_level =  ['CJT_GESAMTTYP', 'LP_FAMILIE_FEIN', 'LP_FAMILIE_GROB', 'LP_STATUS_FEIN', 
                    'LP_STATUS_GROB', 'NATIONALITAET_KZ', 'SHOPPER_TYP', 'GEBAEUDETYP',
                    'CAMEO_DEUG_2015', 'CAMEO_DEU_2015'] 
    
    #-------------------------------------------------------------------------------------------------------------------
    #Re-Encoding non-numerical variables
    df['OST_WEST_KZ'].replace('O','0', inplace=True)
    df['OST_WEST_KZ'].replace('W','1', inplace=True)
    df['OST_WEST_KZ'].unique()

    #-------------------------------------------------------------------------------------------------------------------
    
    #Replacing codes with values from the data dictionary
    for key in list(DictAttr['NATIONALITAET_KZ'].keys()):
        df['NATIONALITAET_KZ'] = df['NATIONALITAET_KZ'].replace(float(key),DictAttr['NATIONALITAET_KZ'][key])
        
    #Creating dummies for the column features
    df = pd.get_dummies(df, columns=['NATIONALITAET_KZ'] , dummy_na=True) 

    #Assiging nan values to the rows of the dummies variables
    NaNindexs = df[df['NATIONALITAET_KZ_nan'] == 1].index
    columns = ['NATIONALITAET_KZ_German-sounding', 'NATIONALITAET_KZ_assimilatednames', 'NATIONALITAET_KZ_foreign-sounding']
    df[columns[0]].loc[NaNindexs] = np.nan
    df[columns[1]].loc[NaNindexs] = np.nan
    df[columns[2]].loc[NaNindexs] = np.nan

    #Dropping the nan dummie variable created
    df.drop('NATIONALITAET_KZ_nan', inplace=True, axis=1)

    #Renaming the created dummies variables
    df = df.rename(columns={columns[0]: 'NAT_KZ_GERSound', columns[1]: 'NAT_KZ_AssimNames', columns[2]:'NAT_KZ_ForeignSound'})

    #-------------------------------------------------------------------------------------------------------------------
    #Dropping the rest of the multi-level categorical features

    del multi_level[5]

    for col in multi_level:
        df.drop(col, inplace=True, axis=1)
        feat_df.drop(feat_df[feat_df['missing_or_unknown'] == ''].index, inplace=True)

    #-------------------------------------------------------------------------------------------------------------------
    #Inserting column for the categorical binary data#
    df.insert(loc=5, column='PREAGENDE_Mainstream', value=df.PRAEGENDE_JUGENDJAHRE.values)
    
    #Generating the content for the 'PRAEGENDE_JUGENDJAHRE' and 'PREAGENDE_Mainstream' columns
    Movement = []
    for key in list(DictAttr['PRAEGENDE_JUGENDJAHRE'].keys()):
        loc = DictAttr['PRAEGENDE_JUGENDJAHRE'][key].find(',')
        Vals = DictAttr['PRAEGENDE_JUGENDJAHRE'][key][0:loc].split('(')

        if len(Vals)>1:
            df['PRAEGENDE_JUGENDJAHRE'] = df['PRAEGENDE_JUGENDJAHRE'].replace(float(key),Vals[0])
            df['PREAGENDE_Mainstream'] = df['PREAGENDE_Mainstream'].replace(float(key),Vals[1])

    #Re-encoding the Mainstream / Avantgarde column for numerical values
    df['PREAGENDE_Mainstream'].replace('Mainstream', 1, inplace=True)
    df['PREAGENDE_Mainstream'].replace('Avantgarde', 0, inplace=True)
    
    #Convert the 'PRAEGENDE_JUGENDJAHRE' column into dummies variables
    df = pd.get_dummies(df, columns=['PRAEGENDE_JUGENDJAHRE'] , dummy_na=True)
    
    
    #Converting unknow values into nan values and dropping the generated nan dummy variable
    NaNindexs = df[df['PRAEGENDE_JUGENDJAHRE_nan'] == 1].index
    columns = df.columns[48:]
    for column in columns:
        df[column].loc[NaNindexs] = np.nan
    df.drop('PRAEGENDE_JUGENDJAHRE_nan', inplace=True, axis=1)    
    
    #Renaming generated dummiest variables
    NewColumns = ['40sReconstYears', '40sWarYears', '50sEconomicMiracle', '50sMilkBar/Ind',
                 '60sEconomicMiracle', '60sGen68', '60sOppToBuildWall', '70sFamOrient', '70sPeaceMov',
                 '80sFDL/CommunistParty',  '80sGenGolf', '80sSwrodsIntoPloughshares',
                 '80sEcologicalAwareness', '90sDigMediaKits', '90sEcologicalAwareness']

    for i, NewCol in enumerate(NewColumns):
        df = df.rename(columns={columns[i]: NewCol})

    #-------------------------------------------------------------------------------------------------------------------
    #Inserting column for the categorical binary data
    df.insert(loc=20, column='CAMEOINTL_Wealthy', value=df.CAMEO_INTL_2015.values)
    
    Movement = []
    for key in list(DictAttr['CAMEO_INTL_2015'].keys()):
        loc = DictAttr['CAMEO_INTL_2015'][key].find('-')
        Vals = DictAttr['CAMEO_INTL_2015'][key].split('s-')

        if key not in ['XX', '-1']:

            if len(Vals)>1:
                df['CAMEOINTL_Wealthy'] = df['CAMEOINTL_Wealthy'].replace(key,int(int(key)/10))
                df['CAMEO_INTL_2015'] = df['CAMEO_INTL_2015'].replace(key,int(key)-int(int(key)/10)*10)

    df = df.rename(columns={'CAMEO_INTL_2015': 'CAMEO_INTL_LifeStageTypology'})

    #-------------------------------------------------------------------------------------------------------------------
    #Dropping the rest of the mixed columns
    Mixed = []

    for col in df.columns:

        if col in feat_df.attribute.values:
            Index = feat_df[feat_df.attribute == col].index.values[0]
            if feat_df.type.loc[Index] == 'mixed':
                Mixed.append(col)

    for col in Mixed:
        df.drop(col, inplace=True, axis=1)
        
    #*******************************************************************************************************************
    # Return the cleaned dataframe.
    return df
        