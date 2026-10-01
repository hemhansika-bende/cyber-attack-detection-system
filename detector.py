import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from tensorflow.keras import layers, models

TRAIN_FILE = "data/UNSW_NB15_training-set.csv"
TEST_FILE = "data/UNSW_NB15_testing-set.csv"

def load_unsw_dataset():
    try:
        train_data = pd.read_csv(TRAIN_FILE)
        test_data = pd.read_csv(TEST_FILE)

        print("Training dataset loaded:", train_data.shape)
        print("Testing dataset loaded:", test_data.shape)

        return train_data, test_data

    except FileNotFoundError as error:
        print("Dataset file was not found.")
        print(error)
        raise

    except Exception as error:
        print("Error while loading the dataset.")
        print(error)
        raise

def preprocess_network_data(train_data, test_data):

    y_train = train_data["label"].astype(int)
    y_test = test_data["label"].astype(int)

    X_train = train_data.drop(
        columns=["label", "attack_cat"],
        errors="ignore"
    )

    X_test = test_data.drop(
        columns=["label", "attack_cat"],
        errors="ignore"
    )

    X_train = X_train.drop(
        columns=["id"],
        errors="ignore"
    )

    X_test = X_test.drop(
        columns=["id"],
        errors="ignore"
    )

    categorical_features = X_train.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_features = X_train.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features
            ),
            (
                "numerical",
                MinMaxScaler(),
                numerical_features
            )
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    X_train_processed = X_train_processed.astype(np.float32)
    X_test_processed = X_test_processed.astype(np.float32)

    print("Processed training data:", X_train_processed.shape)
    print("Processed testing data:", X_test_processed.shape)

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test
    )

def create_autoencoder(input_size, compressed_size=20):

    input_layer = layers.Input(
        shape=(input_size,)
    )

    encoded_layer = layers.Dense(
        64,
        activation="relu"
    )(input_layer)

    compressed_layer = layers.Dense(
        compressed_size,
        activation="relu",
        name="network_feature_layer"
    )(encoded_layer)

    decoded_layer = layers.Dense(
        64,
        activation="relu"
    )(compressed_layer)

    output_layer = layers.Dense(
        input_size,
        activation="sigmoid"
    )(decoded_layer)

    autoencoder = models.Model(
        input_layer,
        output_layer
    )

    encoder = models.Model(
        input_layer,
        compressed_layer
    )

    autoencoder.compile(
        optimizer="adam",
        loss="mse"
    )

    return autoencoder, encoder

def train_network_encoder(X_train):

    input_size = X_train.shape[1]

    autoencoder, encoder = create_autoencoder(
        input_size=input_size,
        compressed_size=20
    )

    print("\nTraining autoencoder...")

    autoencoder.fit(
        X_train,
        X_train,
        epochs=30,
        batch_size=256,
        validation_split=0.2,
        verbose=1
    )

    return encoder

def create_attack_classifier():

    random_forest = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    svm_classifier = SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )

    base_classifiers = [
        ("random_forest", random_forest),
        ("svm", svm_classifier)
    ]

    meta_classifier = LogisticRegression(
        max_iter=1000
    )

    attack_classifier = StackingClassifier(
        estimators=base_classifiers,
        final_estimator=meta_classifier,
        passthrough=True,
        n_jobs=-1
    )

    return attack_classifier

def evaluate_attack_detection(
    classifier,
    X_test,
    y_test
):

    predictions = classifier.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\length================================")
    print("CYBER ATTACK DETECTION RESULTS")
    print("================================")

    print(
        f"\nDetection Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Normal",
                "Attack"
            ]
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

def main():

    print("UNSW-NB15 Cyber Attack Detection")
    print("--------------------------------")

    print("\nLoading network traffic data...")

    train_data, test_data = load_unsw_dataset()

    print("\nPreparing network traffic features...")

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = preprocess_network_data(
        train_data,
        test_data
    )

    print("\nCreating compressed network features...")

    encoder = train_network_encoder(
        X_train
    )

    print("\nEncoding training traffic...")

    X_train_encoded = encoder.predict(
        X_train,
        verbose=0
    )

    print("Encoding testing traffic...")

    X_test_encoded = encoder.predict(
        X_test,
        verbose=0
    )

    print(
        "\nOriginal feature count:",
        X_train.shape[1]
    )

    print(
        "Compressed feature count:",
        X_train_encoded.shape[1]
    )

    print(
        "\nTraining Random Forest + SVM "
        "stacking classifier..."
    )

    attack_classifier = create_attack_classifier()

    attack_classifier.fit(
        X_train_encoded,
        y_train
    )

    print("\nEvaluating attack detection...")

    evaluate_attack_detection(
        attack_classifier,
        X_test_encoded,
        y_test
    )

    print("\nDetection pipeline completed.")

if __name__ == "__main__":
    main()
