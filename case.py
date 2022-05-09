# -*- coding: utf-8 -*-
"""
Created on Sat May  7 06:54:25 2022

@author: tobias Fog
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

'''Load data - data står som komma sepereret i en enkelt kolonne, derfor split 
både data og kolonner på komma'''

df_temp = pd.read_excel(r'/Users/tobiasfog/Dropbox/Mac/Downloads/DataScientist-Case-Dataset.xlsx')

df = df_temp.iloc[:, 0].str.split(',', expand=True)
df.columns = [n.replace('"', '') for n in df_temp.columns.str.split(',')[0]]

'''Exploratory data analysis
customer id er ikke brugbart i analysen(Ville være brugbar hvis analysen havde 
et prædiktivt formål til at kunne identificere kunden efterfølgende)

Konto nummerr omdannes til binær for hvorvidt konto nr er registeret - den oprindelige
kolonne credit_account_id skal derfor fjernes sammen med customer id'''

print('unique values in features:')
[print(set(df[x])) for x in df.columns]



cols_to_remove = ['customer_id', 'credit_account_id']

missing_acc_id = '9b2d5b4678781e53038e91ea5324530a03f27dc1d0e5f6c9bc9d493a23be9de0'

print('')
print('Antal observationer hvor account id mangler:')
print(len([x for x in df['credit_account_id'] if x == missing_acc_id ]))
print('Antal observationer hvor account id ikke mangler:')
print(len([x for x in df['credit_account_id'] if x != missing_acc_id ]))

df['acc_id_exist'] = [1 if x != missing_acc_id else 0 for x in df['credit_account_id']]

df.drop(columns = cols_to_remove, inplace=True, axis=1)
print('')
print('Datatype pr feature:')
print('col datatypes:', df.dtypes)

'''Data cleaning - alle kolonner er strings
to_numeric konverterer kolonnerne hvis de burde være float eller int'''

for col in [col for col in df.columns if col not in ['gender', 'branch']]:
    df[col] = pd.to_numeric(df[col])

print('')
print('Datatype pr feature:')
print('col datatypes:', df.dtypes)
print('')
print('unikke værdier pr feature:')
[print( col,':', df[col].nunique()) for col in df.columns]

    
print('')
print('Missing values pr feature:')
print(df.isnull().sum())

'''Checker om der er systematisk forskel på alder på konvertering. Det er der.'''
print('')
print('missing converters:    ',len([x for index, x in df.iterrows() if pd.isna(x['age']) if x['converted'] == 1]))

print('missing not converters:', len([x for index, x in df.iterrows() if pd.isna(x['age']) if x['converted'] != 1]))

'''
Hvad der skal gøres må vurderes ud fra fordeling af alder på konvertering,
 for nu laves der en binær indikator for rækkerne'''

df['age_missing'] = [1 if pd.isna(x) else 0 for x in df['age']]

'''Plotting '''

sns.countplot(x='converted', data=df)

fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
sns.countplot(ax = axs[0,0],x='customer_segment', hue='converted',data=df)
sns.countplot(ax = axs[0,1],x='gender',hue='converted',data=df)
sns.countplot(ax = axs[1,0],x='related_customers',hue='converted',data=df)
sns.countplot(ax = axs[1,1],x='branch',hue='converted',data=df)

fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
sns.histplot(ax = axs[0,0],x='age', hue='converted',data=df, bins = 10)
sns.countplot(ax = axs[0,1],x='family_size',hue='converted',data=df)
sns.histplot(ax = axs[1,0],x='initial_fee_level',hue='converted',data=df, bins = 10)
sns.countplot(ax = axs[1,1],x='acc_id_exist',hue='converted',data=df)

'''Alder virker ikke til at have store forskelle så missing sættes til median, 
for at kunne bruge informationen om missing alder senere. median alderen er 28.'''

df['age'] = df['age'].fillna(df['age'].median())

'''Tjekker de binære variable for missing, og variablen for alder hvor missing
har fået medianen som værdi'''
#%%
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
sns.countplot(ax = axs[0],x='age_missing', hue='converted',data=df)
sns.histplot(ax = axs[1],x='age', hue='converted',data=df, bins = 10)

#%%

'''Tjekker age_missing som andel af de 28 årige, og de udgør naturligt nok en stor del. 
Potentiel fejlkilde som kan genovervejes hvis age_missing ikke rigtig tyder på at være vigtig.'''

print('')
print('age_missing among 28 years:    ',len([x for index, x in df.iterrows() if x['age'] == 28 if x['age_missing'] == 1]))

print('not age_missing among 28 years:', len([x for index, x in df.iterrows() if x['age'] == 28 if x['age_missing'] != 1]))

'''Logistisk regression kræver encoding af strings
gender bliver derfor binær for hvorvidt køn er woman
branch bliver en række binærer for hvorvidt branchen er den pågældende string.
Der er missing i branch, men vil ikke behandles yderligere,
da de i praksis får samme betydning som den binærer for missing alder'''

df['woman'] = [1 if x == 'female' else 0 for x in df['gender']]

for branch in [x for x in set(df['branch'])]:
    df[f'branch - {branch}'] = [1 if x == f'{branch}' else 0 for x in df['branch']]

'''Omdøber missing branch'''

df['branch - missing'] = df['branch - ']

'''Der er ikke brug for følgende variable længere'''

df.drop(columns = ['gender', 'branch', 'branch - '], inplace=True, axis=1)

'''For at finde de vigtigste parametre til at prædiktere konvertering anvendes 
både en parameterbaseret model og en ikke-parameter model. Først trænes og 
evalueres modellerne på det fulde feature set. Herfra findes et subset som tyder
på at være de vigtigste parametre. Modellerne vil så blive trænet på hver deres 
subset af variable for at se hvor stor præcision subsetet kan opnå. 
 
To ting er vigtige - hvor stor præcision kan subsettet opnå sammenlignet med
det fulde feature set for hver model? Og hvor stort overlap er der mellem subset
på tværs af modeller? Jo bedre præcision og jo mere sammenlignelig, jo mere
sikker er konklusionen om at subsettet er de vigtigste parametre for at forudsige
konvertering.'''

'''split i trænings og test sæt'''

features = [col for col in df.columns if col != 'converted']

X_train, X_test, y_train, y_test = train_test_split(df[features], df['converted'], 
                                                    test_size=.2, random_state=42)

'''træn en parameterbaseret model gennem logistisk regression, og print koefficienterne'''

log_model = LogisticRegression(random_state=42).fit(X_train, y_train)

log_importance = dict(zip(features, log_model.coef_[0]))

[print(k, v) for k, v in log_importance.items()]

log_pred = log_model.predict(X_test)
print('')
print('logit accuracy for full feature set:    ',accuracy_score(y_test, log_pred))

'''Gentag for subset'''

log_subset = ['acc_id_exist', 'woman']

log_model_subset = LogisticRegression(random_state=42).fit(X_train[log_subset], y_train)

log_subset_pred = log_model_subset.predict(X_test[log_subset])

print('logit accuracy for subset feature set:    ', accuracy_score(y_test, log_subset_pred))
print('')

'''Træn en ikke parameterbaseret model genne en Gradient booster og print importance'''

model = XGBClassifier()
n_estimators = [100, 200]
max_depth = [1, 2, 3, 5, 8, 10]
learning_rate=[0.05,0.1, .01]
min_child_weight=[1,2,3,4]

hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight
    }

random_cv = RandomizedSearchCV(estimator=model,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)

random_cv.fit(X_train, y_train)

xgb = random_cv.best_estimator_

xgb.fit(X_train,y_train)

xgb_pred = xgb.predict(X_test)
print('')
print('xgb accuracy for full feature set:    ',accuracy_score(y_test, xgb_pred))
print('')

xgb_importance = dict(zip(features, xgb.feature_importances_))
  
[print(k, v) for k, v in xgb_importance.items()]
    
'''Gentag for subset'''

xgb_subset = ['customer_segment','acc_id_exist', 'woman']

xgb_model_subset = random_cv.best_estimator_

xgb_model_subset.fit(X_train[xgb_subset],y_train)

xgb_subset_pred = xgb_model_subset.predict(X_test[xgb_subset])
print('')
print('xgb accuracy for subset feature set:    ', accuracy_score(y_test, xgb_subset_pred))

'''Tester om xgb er bedre uden customer segment'''

xgb_subset2 = ['acc_id_exist', 'woman']

xgb_model_subset2 = random_cv.best_estimator_

xgb_model_subset2.fit(X_train[xgb_subset2],y_train)

xgb_subset_pred2 = xgb_model_subset2.predict(X_test[xgb_subset2])

print('xgb accuracy for subset2 feature set:    ', accuracy_score(y_test, xgb_subset_pred2))

'''Til sidst afrapporteres auc for samtlige modeller da det er et mål for hvor 
gode modellerne er til at prædiktere positiv klassen.'''

print('')
print('logistisk regression:')
print('auc for full feature set:',roc_auc_score(y_test, log_pred))
print('auc for subset featur sete:',roc_auc_score(y_test, log_subset_pred))
print('')
print('Gradient booster:')
print('auc for full feature set:', roc_auc_score(y_test, xgb_pred))
print('auc for subset feature set:',roc_auc_score(y_test, xgb_subset_pred))
print('auc for subset2 feature set:',roc_auc_score(y_test, xgb_subset_pred2))


'''Konklusion: Der kan prædikteres 78% korrekt ved kun at kigge på køn og hvorvidt konto nr mangler.

Resultatet er umiddelbart ret robust givet at det er konsistent på tværs af learner,
og et meget lille fald i performance for begge learners når der kun trænes på subset.
 
Det er interessant at se hvordan auc falder drastisk ved at tage den umiddelbart
relevante feature customer_segmet med i xgb subset2.

Uanset hvad konvertering betyder så er succesraten størst hvis observationen er
en kvinde som har oplyst konto nr,og lavest hvis det er en mand som ikke har 
oplyst konto nr.'''
 

'''Laver tabeller til latex'''

columns = ['', 'accuracy', 'auc']
model_features = [['Logistisk regression', '',''],
                   ['Full feature set', accuracy_score(y_test, log_pred), roc_auc_score(y_test, log_pred)],
                   ['Subset feature set', accuracy_score(y_test, log_subset_pred),roc_auc_score(y_test, log_subset_pred)],
                     ['Gradient booster', '',''],
                     ['Full feature set',  accuracy_score(y_test, xgb_pred), roc_auc_score(y_test, xgb_pred)],
                     ['Subset feature set', accuracy_score(y_test, xgb_subset_pred), roc_auc_score(y_test, xgb_subset_pred)],
                     ['Subset2 feature set' ,accuracy_score(y_test, xgb_subset_pred2),roc_auc_score(y_test, xgb_subset_pred2)]
                ]

tabel = pd.DataFrame(model_features, columns = columns)

tabel_latex = tabel.to_latex()

logit_coef_list = []
logit_coef_cols = ['feature', 'coefficient']
[logit_coef_list.append([k, v]) for k, v in log_importance.items()]
    
logit_coef = pd.DataFrame(logit_coef_list, columns = logit_coef_cols)
logit_coef = logit_coef.sort_values(by=['coefficient'],  ascending=False)

logit_coef_latex = logit_coef.to_latex()

xgb_importance_list =[]
xgb_importanace_cols = ['feature', 'importance']
[xgb_importance_list.append([k, v]) for k, v in xgb_importance.items()]
    
xgb_importance = pd.DataFrame(xgb_importance_list, columns = xgb_importanace_cols)

xgb_importance = xgb_importance.sort_values(by=['importance'],  ascending=False)

xgb_importance_latex = xgb_importance.to_latex()