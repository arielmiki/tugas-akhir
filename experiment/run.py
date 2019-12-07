import os, shutil, sys

if len(sys.argv) < 5:
    print("Usage: [DATA PATH|STRING] [NODE COUNT|INT] [ACCURACY THRESHOLD|FLOAT] [OUTPUT FOLDER|STRING]")
    exit()
    
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from keras import models, layers
from implementation.abduction import MultilayerRELUEXplanation
import pickle
from timeit import default_timer as timer
import math
    
# overwrite with sys argv
data_source = sys.argv[1]
node_count = int(sys.argv[2])
accuracy_threshold = float(sys.argv[3])
output_folder = sys.argv[4]
verbose = len(sys.argv) > 5

if os.path.exists(output_folder) and os.path.isdir(output_folder):
    shutil.rmtree(output_folder)
os.mkdir(output_folder)

df = pd.read_csv(data_source)
df = df.sample(frac=1, random_state=42)
X = df.drop('label', axis=1).values
y = df['label'].values.reshape(-1,1)

ohe = OneHotEncoder(sparse=False, categories='auto')
y_transform = ohe.fit_transform(y)

ss = StandardScaler()
X_norm = ss.fit_transform(X)

accuracy = 0
while accuracy < accuracy_threshold:
    model = models.Sequential()
    model.add(layers.Dense(node_count, input_dim=X_norm.shape[1], activation='relu'))
    model.add(layers.Dense(y_transform.shape[1], activation='relu'))
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(X_norm, y_transform, epochs=1000, batch_size=32, verbose=0)
    y_pred = model.predict(X_norm)
    accuracy = accuracy_score(np.argmax(y_pred, axis=1), np.argmax(y_transform, axis=1))
    print("Accuracy: %f" % accuracy)

shutil.rmtree(output_folder)
os.mkdir(output_folder)

with open(output_folder + "/accuracy.info", 'w') as f:
    print(accuracy, file=f)

with open(output_folder + "/model.pickle", 'wb') as pfile:
    pickle.dump((model, ss, ohe), pfile, protocol=pickle.HIGHEST_PROTOCOL)
    
    
relu_explanation = MultilayerRELUEXplanation(model.get_weights(), np.max(y_pred),  problem_name=output_folder + "/relu-encoding")


result = pd.DataFrame(X_norm)
result.columns  = df.columns.drop('label')
result['y_actual'] = np.argmax(y_transform, axis=1)
result['y_pred'] = np.argmax(y_pred, axis=1)


def compute_explanation(df, function, timeout = 1800, verbose=False):
    print('Process will timeout after %d second' % timeout)
    start = timer()
    explanation_result = []
    time_result = []
    for index, row in df.iterrows():
        elapsed = timer() - start
        if verbose:
            print('Elapsed %d: %f' % (index, elapsed))
        if  elapsed < timeout:
            t = timer()
            explanation_result.append(function(row))
            time_result.append(timer() - t)
        else:
            print('TIMEOUT: elapsed %f' % elapsed)
            return (pd.Series(), pd.Series())
    print('COMPLETE: elapsed: %f' % elapsed)
    return (explanation_result, time_result)  

def count(x):
    try:
        return len(x)
    except:
        return 0

print("Computing subset minimal")
## Get Subset Minimal
get_subset_minimal = lambda row: relu_explanation.get_subset_minimal(row[:X.shape[1]], row['y_pred'])
computed_result = compute_explanation(result, get_subset_minimal, verbose=verbose)
result['MinSub'] = computed_result[0]
result['MinSubTime'] = computed_result[1]
result['MinSubCount'] = result['MinSub'].apply(count)
result.to_csv(output_folder + '/result.csv',index=False)

print("Computing randomized subset minimal (1)")
## Get Randomized Subset Minimal 1
get_randomized_1 = lambda row: relu_explanation.get_subset_minimal_with_randomized(row[:X.shape[1]], row['y_pred'])
computed_result =  compute_explanation(result, get_randomized_1, verbose=verbose)
result['MinSubRnd1'] = computed_result[0]
result['MinSubRnd1Time'] = computed_result[1]
result['MinSubRnd1Count'] = result['MinSubRnd1'].apply(count)

print("Computing randomized subset minimal (3)")
## Get Randomized Subset Minimal 3
get_randomized_3 = lambda row: relu_explanation.get_subset_minimal_with_randomized(row[:X.shape[1]], row['y_pred'], 3)
computed_result =  compute_explanation(result, get_randomized_3, verbose=verbose)
result['MinSubRnd3'] = computed_result[0]
result['MinSubRnd3Time'] = computed_result[1]
result['MinSubRnd3Count'] = result['MinSubRnd3'].apply(count)

print("Computing randomized subset minimal (5)")
## Get Randomized Subset Minimal 5
get_randomized_5 = lambda row: relu_explanation.get_subset_minimal_with_randomized(row[:X.shape[1]], row['y_pred'], 5)
computed_result =  compute_explanation(result, get_randomized_5, verbose=verbose)
result['MinSubRnd5'] = computed_result[0]
result['MinSubRnd5Time'] = computed_result[1]
result['MinSubRnd5Count'] = result['MinSubRnd5'].apply(count)
result.to_csv(output_folder + '/result.csv',index=False)

print("Computing cardinality minimal")
## Get Cardinality Minimal
get_cardinality_minimal = lambda row: relu_explanation.get_cardinality_minimal(row[:X.shape[1]], row['y_pred'])
computed_result =  compute_explanation(result, get_cardinality_minimal, verbose=verbose)
result['MinCard'] = computed_result[0]
result['MinCardTime'] = computed_result[1]
result['MinCardCount'] = result['MinCard'].apply(count)
result.to_csv(output_folder + '/result.csv',index=False)