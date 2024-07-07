import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def cargar_datos(archivo):
    df = pd.read_csv(archivo)
    return df

def preprocesar_datos(df):
    df['LeaveOrNot'] = df['LeaveOrNot'].map({0: 'Not Leave', 1: 'Leave'})
    df = df.dropna(subset=['ExperienceInCurrentDomain', 'JoiningYear'])
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['PaymentTier'].fillna(df['PaymentTier'].mode()[0], inplace=True)
    for col in ['Age', 'ExperienceInCurrentDomain', 'JoiningYear']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
    df['CategoriaEdad'] = pd.cut(df['Age'], bins=[0, 12, 19, 39, 59, 100], labels=['Niño', 'Adolescente', 'Joven Adulto', 'Adulto', 'Adulto Mayor'])
    return df

def clasificar_datos():
    df = cargar_datos('empleados.csv')
    df = preprocesar_datos(df)

    X = df.drop(columns=['LeaveOrNot', 'CategoriaEdad'])
    y = df['LeaveOrNot']
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Random Forest sin cambios
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # Random Forest con class_weight
    rf_balanced = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf_balanced.fit(X_train, y_train)
    y_pred_rf_balanced = rf_balanced.predict(X_test)

    # Métricas
    print(f'Accuracy (RF): {accuracy_score(y_test, y_pred_rf):.2f}')
    print(f'F1 Score (RF): {f1_score(y_test, y_pred_rf, pos_label="Leave"):.2f}')
    print(f'Accuracy (RF Balanced): {accuracy_score(y_test, y_pred_rf_balanced):.2f}')
    print(f'F1 Score (RF Balanced): {f1_score(y_test, y_pred_rf_balanced, pos_label="Leave"):.2f}')

    # Matrices
    cm_rf = confusion_matrix(y_test, y_pred_rf, labels=['Not Leave', 'Leave'])
    disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Not Leave', 'Leave'])
    disp_rf.plot(cmap=plt.cm.Blues)
    plt.title('Matriz de Confusión (Random Forest)')
    plt.show()

    cm_rf_balanced = confusion_matrix(y_test, y_pred_rf_balanced, labels=['Not Leave', 'Leave'])
    disp_rf_balanced = ConfusionMatrixDisplay(confusion_matrix=cm_rf_balanced, display_labels=['Not Leave', 'Leave'])
    disp_rf_balanced.plot(cmap=plt.cm.Blues)
    plt.title('Matriz de Confusión (Random Forest Balanced)')
    plt.show()

if __name__ == "__main__":
    clasificar_datos()
