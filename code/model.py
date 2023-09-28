# Desarrollo de la sección 3. Ahora con otra red y datos

# Importar los paquetes requeridos
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.model_selection import train_test_split
from pgmpy.inference import VariableElimination
# Definir el path en dónde se encuentran los datos
path_datos = 'C:/Users/berna/OneDrive/Escritorio/Universidad de los Andes/Semestre 2023-2/Análitica Computacional para la Toma de Decisiones/Proyecto/predict+students+dropout+and+academic+success'


# Cargar los datos
data = pd.read_csv(path_datos+'/data.csv', delimiter=";")
# For numerical columns, fill NaN with mean
for col in data.select_dtypes(include=['float64', 'int64']):
    data[col].fillna(data[col].mean(), inplace=True)

# For categorical columns, fill NaN with mode
for col in data.select_dtypes(include=['object']):
    data[col].fillna(data[col].mode()[0], inplace=True)


# Exploración de los datos
data.head()
data['Curricular units 1st sem (enrolled)'].value_counts()
data=data[data['Curricular units 1st sem (enrolled)']!=0]
data['perc_approved_sem1'] = data['Curricular units 1st sem (approved)']/data['Curricular units 1st sem (enrolled)']
data['perc_approved_sem2'] = data['Curricular units 2nd sem (approved)']/data['Curricular units 2nd sem (enrolled)']

nan_summary = data.isna().sum()
print(nan_summary)

nan_summary = nan_summary[nan_summary > 0]
print(nan_summary)

print(data)

# Partir los datos en entrenamiento y prueba
train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)


# Definir la red bayesiana
model = BayesianNetwork([("Unemployment rate", "Debtor"), ("Inflation rate", "Debtor"),
                         ("Debtor", "perc_approved_sem1"), ("Scholarship holder", "perc_approved_sem1"),
                         ("perc_approved_sem1", "perc_approved_sem2"), ("perc_approved_sem2","Target"),
                         ("Age at enrollment","Target")])

model.fit(data=train_data, estimator=MaximumLikelihoodEstimator)
data.columns
for i in model.nodes():
    print(model.get_cpds(i))

# Predict probabilities for testing

# Initialize VariableElimination class with the model
inference = VariableElimination(model)
test_data.shape
# For each row in the test_data, predict the probability of "lung"
test_data.head()
target_probabilities = []
row=test_data.iloc[0,:]
for index, row in test_data.iterrows():
    prob = inference.query(variables=["Target"], evidence={"Age at enrollment": row["Age at enrollment"],
                                                          "perc_approved_sem2": row['perc_approved_sem2']})
    target_probabilities.append(prob.values[1])


# Print the probabilities

prob_lung_cancer=[]
for prob in lung_probabilities:
    value_prob_cancer=prob.values[1]
    prob_lung_cancer.append(value_prob_cancer)

test_data['lung'] = test_data['lung'].map({'yes': 1, 'no': 0})

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
        "Exactitud": accuracy,
        "Precisión": precision,
        "Recall": recall,
        "Especificidad": specificity,
        "Verdaderos positivos": TP,
        "Falsos positivos": FP,
        "Verdaderos negativos": TN,
        "Falsos negativos": FN
    }



import pandas as pd

# Convert the list of probabilities into a pandas DataFrame
prob_df = pd.DataFrame(prob_lung_cancer, columns=['Probability'])

# Get descriptive statistics
desc_stats = prob_df.describe()

print(desc_stats)
train_data['lung'] = train_data['lung'].map({'yes': 1, 'no': 0})

quantile_cutoff = 1 - train_data['lung'].mean()
print(quantile_cutoff)


cutoff = prob_df['Probability'].quantile(quantile_cutoff)
# Dado que en train el 94.468% de las personas no tienen cancer de Pulmón
# se predecirá que tiene cancer de Pulmón si la probabilidad 
# está por encima del percentil 94.668 de las probabilidades.
# Esto no está escrito, solo es una forma de definir el punto de corte.
# Esto de alguna forma que los casos de cancer se distribuyen de igual manera
# en Training y Test

# Métricas de desempeño
threshold = cutoff  # Puedes cambiar este valor según tus necesidades
performance = evaluate_performance(prob_lung_cancer, test_data['lung'], threshold)

for key, value in performance.items():
    print(f"{key}: {value}")

