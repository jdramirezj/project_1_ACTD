# Desarrollo de la sección 3. Ahora con otra red y datos

# Importar los paquetes requeridos
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.model_selection import train_test_split
from pgmpy.inference import VariableElimination
import numpy as np

# Definir el path en dónde se encuentran los datos
path_datos_samuel = 'C:/Users/berna/OneDrive/Escritorio/Universidad de los Andes/Semestre 2023-2/Análitica Computacional para la Toma de Decisiones/Proyecto/predict+students+dropout+and+academic+success'
path_datos_juan = '/Users/juandramirezj/Documents/Universidad - MIIND/ACTD/proyecto_1/project_1_ACTD/data'

# Cargar los datos
data = pd.read_csv(path_datos_samuel+'/data.csv', delimiter=";")
# For numerical columns, fill NaN with mean
for col in data.select_dtypes(include=['float64', 'int64']):
    data[col].fillna(data[col].mean(), inplace=True)

# For categorical columns, fill NaN with mode
for col in data.select_dtypes(include=['object']):
    data[col].fillna(data[col].mode()[0], inplace=True)

# Exploración de los datos
data.head()
data=data[data['Curricular units 1st sem (enrolled)']!=0]
data['perc_approved_sem1'] = data['Curricular units 1st sem (approved)']/data['Curricular units 1st sem (enrolled)']
data['perc_approved_sem2'] = data['Curricular units 2nd sem (approved)']/data['Curricular units 2nd sem (enrolled)']



nan_summary = data.isna().sum()
print(nan_summary)

nan_summary = nan_summary[nan_summary > 0]
print(nan_summary)

print(data)

# Discretize variables and 'perc_approved_sem2' into quartiles
data['Inflation rate'] = pd.qcut(data['Inflation rate'], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
data['Unemployment rate'] = pd.qcut(data['Unemployment rate'], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
data['perc_approved_sem1'] = pd.qcut(data['perc_approved_sem1'], q=2, labels=["Q1", "Q2"])
data['perc_approved_sem2'] = pd.qcut(data['perc_approved_sem2'], q=2, labels=["Q1", "Q2"])
data['Age at enrollment'] = pd.qcut(data['Age at enrollment'], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
data['Admission grade'] = pd.qcut(data['Admission grade'], q=2, labels=["Q1", "Q2"])
data['Previous qualification (grade)'] = pd.qcut(data['Previous qualification (grade)'], q=2, labels=["Q1", "Q2"])
data['Target']
data['actual_target'] = np.where(data['Target']=='Dropout',1,0)

# Partir los datos en entrenamiento y prueba
train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)

# model1
# Definir la red bayesiana
model1 = BayesianNetwork([("Unemployment rate", "perc_approved_sem1"), ("Inflation rate", "perc_approved_sem1"),
                         ("Debtor", "perc_approved_sem1"), ("Scholarship holder", "perc_approved_sem1"),
                         ("perc_approved_sem1", "perc_approved_sem2"), ("perc_approved_sem2","actual_target"),
                         ("Age at enrollment","actual_target")])

model1.fit(data=train_data, estimator=MaximumLikelihoodEstimator)
data.columns
for i in model1.nodes():
    print(model1.get_cpds(i))

# Predict probabilities for testing

# Initialize VariableElimination class with the model
inference = VariableElimination(model1)
test_data.shape
# For each row in the test_data, predict the probability of "lung"
test_data.head()
target_probabilities = []
for index, row in test_data.iterrows():
    prob = inference.query(variables=["actual_target"], evidence={"Age at enrollment": row["Age at enrollment"],
                                                                  "Unemployment rate": row["Unemployment rate"],
                                                                  "Inflation rate": row["Inflation rate"],
                                                                  "Debtor": row["Debtor"],
                                                                  "Scholarship holder": row["Scholarship holder"]})
    target_probabilities.append(prob)


# Print the probabilities: OJO FALTA MIRAR ESTO
target_probabilities[0].values
prob_target_dropout=[]
for prob in target_probabilities:
    value_prob_cancer=prob.values[1]
    prob_target_dropout.append(value_prob_cancer)
# model2
# Definir la red bayesiana
model2 = BayesianNetwork([("Previous qualification (grade)", "Debtor"), ("Previous qualification (grade)", "Scholarship holder"), ("Previous qualification (grade)", "perc_approved_sem1"),
                          ("Admission grade", "Debtor"), ("Admission grade", "Scholarship holder"), ("Admission grade", "perc_approved_sem1"),
                          ("Debtor", "perc_approved_sem2"), ("Scholarship holder", "perc_approved_sem2"), ("perc_approved_sem1", "perc_approved_sem2"),
                          ("perc_approved_sem2","actual_target")])
model2.fit(data=train_data, estimator=MaximumLikelihoodEstimator)
data.columns
for i in model2.nodes():
    print(model2.get_cpds(i))

# Predict probabilities for testing

# Initialize VariableElimination class with the model
inference = VariableElimination(model2)
test_data.shape
# For each row in the test_data, predict the probability of "lung"
test_data.head()
target_probabilities = []
for index, row in test_data.iterrows():
    prob = inference.query(variables=["actual_target"], evidence={"Previous qualification (grade)": row["Previous qualification (grade)"],
                                                                  "Admission grade": row["Admission grade"]})
    target_probabilities.append(prob)


# Print the probabilities
target_probabilities[0].values
prob_target_dropout=[]
for prob in target_probabilities:
    value_prob_cancer=prob.values[1]
    prob_target_dropout.append(value_prob_cancer)
# model3
# Definir la red bayesiana
model3 = BayesianNetwork([("International", "perc_approved_sem1"), ("Marital status", "perc_approved_sem1"), ("Gender", "perc_approved_sem1"),
                          ("Unemployment rate", "perc_approved_sem1"), ("Inflation rate", "perc_approved_sem1"), ("Displaced", "perc_approved_sem1"),
                          ("Educational special needs", "perc_approved_sem1"), ("perc_approved_sem1", "perc_approved_sem2"), ("Scholarship holder", "actual_target"),
                          ("perc_approved_sem2","actual_target"), ("Debtor","actual_target")])
model3.fit(data=train_data, estimator=MaximumLikelihoodEstimator)
data.columns
for i in model3.nodes():
    print(model3.get_cpds(i))

# Predict probabilities for testing

# Initialize VariableElimination class with the model
inference = VariableElimination(model3)
test_data.shape
# For each row in the test_data, predict the probability of "lung"
test_data.head()
target_probabilities = []
for index, row in test_data.iterrows():
    prob = inference.query(variables=["actual_target"], evidence={"International": row["International"],
                                                                  "Marital status": row["Marital status"],
                                                                  "Gender": row["Gender"],
                                                                  "Unemployment rate": row["Unemployment rate"],
                                                                  "Inflation rate": row["Inflation rate"],
                                                                  "Displaced": row["Displaced"],
                                                                  "Educational special needs": row["Educational special needs"],
                                                                  "Scholarship holder": row["Scholarship holder"],
                                                                  "Debtor": row["Debtor"]})
    target_probabilities.append(prob)


# Print the probabilities
target_probabilities[0].values
prob_target_dropout=[]
for prob in target_probabilities:
    value_prob_cancer=prob.values[1]
    prob_target_dropout.append(value_prob_cancer)


# model4
# Definir la red bayesiana
model4 = BayesianNetwork([("International", "Debtor"), ("Displaced", "Debtor"),
                          ("Debtor", "perc_approved_sem2"), ("Marital status", "actual_target"),
                          ("perc_approved_sem2","actual_target")])
model4.fit(data=train_data, estimator=MaximumLikelihoodEstimator)
data.columns
for i in model4.nodes():
    print(model4.get_cpds(i))

# Predict probabilities for testing

# Initialize VariableElimination class with the model
inference = VariableElimination(model4)
test_data.shape
# For each row in the test_data, predict the probability of "lung"
test_data.head()
target_probabilities = []
for index, row in test_data.iterrows():
    prob = inference.query(variables=["actual_target"], evidence={"International": row["International"],
                                                                  "Marital status": row["Marital status"],
                                                                  "Displaced": row["Displaced"]})
    target_probabilities.append(prob)


# Print the probabilities
target_probabilities[0].values
prob_target_dropout=[]
for prob in target_probabilities:
    value_prob_cancer=prob.values[1]
    prob_target_dropout.append(value_prob_cancer)

# model5
# Definir la red bayesiana
model5 = BayesianNetwork([("Age at enrollment", "Admission grade"), ("Educational special needs", "Admission grade"),
                          ("Admission grade", "perc_approved_sem1"), ("Debtor", "actual_target"),
                          ("perc_approved_sem1","actual_target")])
model5.fit(data=train_data, estimator=MaximumLikelihoodEstimator)
data.columns
for i in model5.nodes():
    print(model5.get_cpds(i))

# Predict probabilities for testing

# Initialize VariableElimination class with the model
inference = VariableElimination(model5)
test_data.shape
# For each row in the test_data, predict the probability of "lung"
test_data.head()
target_probabilities = []
for index, row in test_data.iterrows():
    prob = inference.query(variables=["actual_target"], evidence={"Age at enrollment": row["Age at enrollment"],
                                                                  "Educational special needs": row["Educational special needs"],
                                                                  "Debtor": row["Debtor"]})
    target_probabilities.append(prob)


# Print the probabilities
target_probabilities[0].values
prob_target_dropout=[]
for prob in target_probabilities:
    value_prob_cancer=prob.values[1]
    prob_target_dropout.append(value_prob_cancer)

# UNTIL HERE WE ALREADY FOUND PREDICTED PROBAILITIES ON TEST

def evaluate_performance(predictions_prob, true_labels, threshold=0.5):
    # Convert the probabilities into predictions based on the specified threshold
    predictions = [1 if prob >= threshold else 0 for prob in predictions_prob]

    # Compute TP, FP, TN, and FN
    TP = sum([1 for i, j in zip(predictions, true_labels) if i == 1 and j == 1])
    FP = sum([1 for i, j in zip(predictions, true_labels) if i == 1 and j == 0])
    TN = sum([1 for i, j in zip(predictions, true_labels) if i == 0 and j == 0])
    FN = sum([1 for i, j in zip(predictions, true_labels) if i == 0 and j == 1])

    # Compute the metrics
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "True positives": TP,
        "False positives": FP,
        "True negatives": TN,
        "False negatives": FN
    }

# Convert the list of probabilities into a pandas DataFrame
prob_df = pd.DataFrame(prob_target_dropout, columns=['Probability'])

# Get descriptive statistics
desc_stats = prob_df.describe()

print(desc_stats)
quantile_cutoff = 1 - test_data['actual_target'].mean()
print(quantile_cutoff)


cutoff = prob_df['Probability'].quantile(quantile_cutoff)

# Métricas de desempeño
threshold = cutoff  
performance = evaluate_performance(prob_target_dropout, test_data['actual_target'], threshold)

for key, value in performance.items():
    print(f"{key}: {value}")

