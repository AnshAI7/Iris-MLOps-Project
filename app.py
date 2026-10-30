import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Iris AI Classifier",
    page_icon="🌸",
    layout="wide"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: 38px;
    font-weight: bold;
}

div[data-testid="stMetric"] {
    background-color: rgba(80, 130, 180, 0.10);
    padding: 16px;
    border-radius: 12px;
    border: 1px solid rgba(130, 160, 190, 0.2);
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD DATASET
# =====================================================

@st.cache_data
def load_dataset():

    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    df["species"] = [
        iris.target_names[i] for i in iris.target
    ]

    return df, iris


df, iris = load_dataset()

FEATURE_COLUMNS = iris.feature_names

SPECIES_NAMES = list(iris.target_names)


# =====================================================
# IRIS SPECIES IMAGES
# =====================================================

FLOWER_IMAGES = {
    "setosa": (
        "https://commons.wikimedia.org/wiki/Special:FilePath/"
        "Iris_setosa_2.jpg"
    ),

    "versicolor": (
        "https://commons.wikimedia.org/wiki/Special:FilePath/"
        "Iris_versicolor_1.jpg"
    ),

    "virginica": (
        "https://commons.wikimedia.org/wiki/Special:FilePath/"
        "Iris_virginica_-_NRCS.jpg"
    )
}


# =====================================================
# LOAD TRAINED MODEL
# =====================================================

MODEL_PATH = Path(__file__).parent / "iris_rf_model.joblib"


@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


try:

    model = load_model()

except Exception as error:

    st.error("Unable to load the trained model.")

    st.write(
        "Check whether iris_rf_model.joblib exists in the project folder."
    )

    st.exception(error)

    st.stop()


# =====================================================
# SESSION STATE
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "bulk_results" not in st.session_state:
    st.session_state.bulk_results = None

if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"


def go_to_page(page_name):

    st.session_state.page = page_name


# =====================================================
# PREDICTION FUNCTION
# =====================================================

def make_prediction(values):

    input_data = pd.DataFrame(
        [values],
        columns=FEATURE_COLUMNS
    )

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    confidence = float(np.max(probabilities) * 100)

    return prediction, probabilities, confidence


# =====================================================
# SAVE PREDICTION HISTORY
# =====================================================

def save_history(values, prediction, confidence):

    record = {
        "Date & Time": datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        ),
        "Sepal Length (cm)": values[0],
        "Sepal Width (cm)": values[1],
        "Petal Length (cm)": values[2],
        "Petal Width (cm)": values[3],
        "Predicted Species": prediction,
        "Confidence (%)": round(confidence, 2)
    }

    st.session_state.history.append(record)


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("🌸 Iris AI Dashboard")

st.sidebar.caption("Machine Learning Application")

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔍 Flower Prediction",
        "📊 Data Visualization",
        "📈 Model Performance",
        "📜 Prediction History",
        "📁 Bulk Prediction",
        "📘 About Project"
    ],
    key="page"
)

st.sidebar.divider()

st.sidebar.metric(
    "Total Predictions",
    len(st.session_state.history)
)

st.sidebar.caption("Algorithm: Random Forest")

st.sidebar.caption("Dataset: Iris Dataset")


# =====================================================
# HOME PAGE
# =====================================================

if page == "🏠 Home":

    st.title("🌸 Welcome to Iris AI Classifier")

    st.write(
        "An interactive Machine Learning application "
        "that predicts Iris flower species using four measurements."
    )

    st.info(
        "Enter flower measurements and let the trained "
        "Random Forest model predict the species."
    )

    st.divider()

    # IRIS FLOWER INTRODUCTION

    st.subheader("🌺 Discover the Iris Flowers")

    image_columns = st.columns(3)

    for index, species in enumerate(SPECIES_NAMES):

        with image_columns[index]:

            st.image(
                FLOWER_IMAGES[species],
                caption=f"Iris {species.capitalize()}",
                use_container_width=True
            )

    st.divider()

    # PROJECT OVERVIEW

    st.subheader("📊 Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Samples", len(df))

    col2.metric("Flower Species", "3")

    col3.metric("Input Features", "4")

    col4.metric(
        "Your Predictions",
        len(st.session_state.history)
    )

    st.divider()

    # SPECIES DETAILS

    st.subheader("🌸 Explore Iris Flower Species")

    descriptions = {
        "setosa": "Small petals and distinctive flower structure.",
        "versicolor": "Medium-sized flower with violet-blue shades.",
        "virginica": "A larger Iris flower with broad petals."
    }

    columns = st.columns(3)

    for index, species in enumerate(SPECIES_NAMES):

        with columns[index]:

            st.subheader("🌸 " + species.capitalize())

            st.write(descriptions[species])

            st.metric(
                "Dataset Samples",
                int((df["species"] == species).sum())
            )

    st.divider()

    # QUICK ACTIONS

    st.subheader("🚀 Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.button(
            "🔍 Start Prediction",
            use_container_width=True,
            on_click=go_to_page,
            args=("🔍 Flower Prediction",)
        )

    with col2:

        st.button(
            "📊 View Visualization",
            use_container_width=True,
            on_click=go_to_page,
            args=("📊 Data Visualization",)
        )

    with col3:

        st.button(
            "📈 Model Performance",
            use_container_width=True,
            on_click=go_to_page,
            args=("📈 Model Performance",)
        )

    st.divider()

    st.caption(
        "Iris AI Classifier | Machine Learning Project"
    )


# =====================================================
# FLOWER PREDICTION
# =====================================================

elif page == "🔍 Flower Prediction":

    st.title("🔍 Flower Prediction")

    st.write(
        "Enter the four measurements of an Iris flower."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🌿 Sepal Measurements")

        sepal_length = st.slider(
            "Sepal Length (cm)",
            4.0, 8.0, 5.1, 0.1
        )

        sepal_width = st.slider(
            "Sepal Width (cm)",
            2.0, 4.5, 3.5, 0.1
        )

    with col2:

        st.subheader("🌸 Petal Measurements")

        petal_length = st.slider(
            "Petal Length (cm)",
            1.0, 7.0, 1.4, 0.1
        )

        petal_width = st.slider(
            "Petal Width (cm)",
            0.1, 2.5, 0.2, 0.1
        )

    values = [
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]

    st.divider()

    # INPUT SUMMARY

    st.subheader("📋 Input Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Sepal Length", f"{sepal_length:.1f} cm")
    c2.metric("Sepal Width", f"{sepal_width:.1f} cm")
    c3.metric("Petal Length", f"{petal_length:.1f} cm")
    c4.metric("Petal Width", f"{petal_width:.1f} cm")

    # PREDICTION BUTTON

    if st.button(
        "🌸 Predict Flower Species",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Analysing flower measurements..."):

            prediction, probabilities, confidence = make_prediction(values)

        save_history(values, prediction, confidence)

        st.success("Prediction completed successfully!")

        st.divider()

        result1, result2 = st.columns(2)

        with result1:

            st.subheader("🌺 Predicted Species")

            st.title(prediction.capitalize())

            st.metric(
                "Prediction Probability",
                f"{confidence:.2f}%"
            )

            st.progress(
                min(int(round(confidence)), 100)
            )

        with result2:

            st.subheader("🌸 Predicted Iris Flower")

            st.image(
                FLOWER_IMAGES[prediction.lower()],
                caption=f"Iris {prediction.capitalize()}",
                use_container_width=True
            )

        st.divider()

        # SPECIES PROBABILITIES

        st.subheader("📊 Species Probabilities")

        probability_df = pd.DataFrame({
            "Species": model.classes_,
            "Probability (%)": probabilities * 100
        })

        st.bar_chart(
            probability_df.set_index("Species")
        )

        st.caption(
            "Probability is the model's estimated confidence, "
            "not a guarantee of correctness."
        )


# =====================================================
# DATA VISUALIZATION
# =====================================================

elif page == "📊 Data Visualization":

    st.title("📊 Iris Data Visualization")

    st.write(
        "Explore the Iris dataset through different charts "
        "and understand the relationship between flower measurements."
    )

    st.divider()

    # DATASET OVERVIEW

    st.subheader("📋 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df))

    col2.metric("Total Features", len(FEATURE_COLUMNS))

    col3.metric("Species", df["species"].nunique())

    st.divider()

    # DATASET PREVIEW

    st.subheader("🔍 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.divider()

    # SPECIES DISTRIBUTION

    st.subheader("🌺 Species Distribution")

    species_counts = df["species"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        species_counts.index,
        species_counts.values
    )

    ax.set_title("Number of Samples in Each Species")

    ax.set_xlabel("Species")

    ax.set_ylabel("Number of Samples")

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # FEATURE DISTRIBUTION

    st.subheader("📏 Feature Distribution")

    selected_feature = st.selectbox(
        "Select a feature",
        FEATURE_COLUMNS
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    for species in SPECIES_NAMES:

        values = df[
            df["species"] == species
        ][selected_feature]

        ax.hist(
            values,
            alpha=0.5,
            label=species,
            bins=12
        )

    ax.set_title(
        f"Distribution of {selected_feature}"
    )

    ax.set_xlabel("Measurement (cm)")

    ax.set_ylabel("Frequency")

    ax.legend()

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # BOX PLOT

    st.subheader("📦 Feature Comparison")

    selected_box_feature = st.selectbox(
        "Choose feature for comparison",
        FEATURE_COLUMNS,
        key="box_feature"
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    box_data = [
        df[df["species"] == species][selected_box_feature]
        for species in SPECIES_NAMES
    ]

    ax.boxplot(
        box_data,
        labels=SPECIES_NAMES
    )

    ax.set_title(
        f"{selected_box_feature} by Species"
    )

    ax.set_ylabel("Measurement (cm)")

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # CORRELATION HEATMAP

    st.subheader("🔥 Feature Correlation")

    correlation = df[FEATURE_COLUMNS].corr()

    fig, ax = plt.subplots(figsize=(8, 5))

    image = ax.imshow(
        correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    ax.set_xticks(range(len(FEATURE_COLUMNS)))

    ax.set_yticks(range(len(FEATURE_COLUMNS)))

    ax.set_xticklabels(
        FEATURE_COLUMNS,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(FEATURE_COLUMNS)

    for i in range(len(FEATURE_COLUMNS)):

        for j in range(len(FEATURE_COLUMNS)):

            ax.text(
                j,
                i,
                f"{correlation.iloc[i, j]:.2f}",
                ha="center",
                va="center"
            )

    fig.colorbar(image, ax=ax)

    ax.set_title("Correlation Between Features")

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # PCA VISUALIZATION

    st.subheader("🧬 PCA Visualization")

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        df[FEATURE_COLUMNS]
    )

    pca = PCA(n_components=2)

    pca_values = pca.fit_transform(scaled_data)

    pca_df = pd.DataFrame(
        pca_values,
        columns=["PC1", "PC2"]
    )

    pca_df["species"] = df["species"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for species in SPECIES_NAMES:

        selected = pca_df[
            pca_df["species"] == species
        ]

        ax.scatter(
            selected["PC1"],
            selected["PC2"],
            label=species,
            alpha=0.7
        )

    variance = pca.explained_variance_ratio_ * 100

    ax.set_xlabel(
        f"PC1 ({variance[0]:.2f}% variance)"
    )

    ax.set_ylabel(
        f"PC2 ({variance[1]:.2f}% variance)"
    )

    ax.set_title("PCA of Iris Dataset")

    ax.legend()

    st.pyplot(fig)

    plt.close(fig)

    st.info(
        f"Total variance explained by two components: "
        f"{variance.sum():.2f}%"
    )


# =====================================================
# MODEL PERFORMANCE
# =====================================================

elif page == "📈 Model Performance":

    st.title("📈 Model Performance")

    st.write(
        "Evaluate the Random Forest model using "
        "test accuracy, cross-validation and classification metrics."
    )

    st.divider()

    X = df[FEATURE_COLUMNS]

    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # TEST PREDICTIONS

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # CROSS VALIDATION

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    cv_scores = cross_val_score(
        cv_model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    # PERFORMANCE OVERVIEW

    st.subheader("🎯 Performance Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Test Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    col2.metric(
        "Mean CV Accuracy",
        f"{cv_scores.mean() * 100:.2f}%"
    )

    col3.metric(
        "Standard Deviation",
        f"{cv_scores.std() * 100:.2f}%"
    )

    col4.metric(
        "Test Samples",
        len(X_test)
    )

    st.divider()

    # CROSS VALIDATION CHART

    st.subheader("📊 5-Fold Cross Validation")

    cv_df = pd.DataFrame({
        "Fold": [
            "Fold 1",
            "Fold 2",
            "Fold 3",
            "Fold 4",
            "Fold 5"
        ],
        "Accuracy (%)": cv_scores * 100
    })

    st.bar_chart(
        cv_df.set_index("Fold")
    )

    st.dataframe(
        cv_df,
        use_container_width=True
    )

    st.divider()

    # CLASSIFICATION REPORT

    st.subheader("📋 Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    st.divider()

    # CONFUSION MATRIX

    st.subheader("🎯 Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=model.classes_
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    display_matrix = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_
    )

    display_matrix.plot(
        cmap="Blues",
        values_format="d",
        ax=ax
    )

    ax.set_title("Random Forest Confusion Matrix")

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # FEATURE IMPORTANCE

    st.subheader("🌳 Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": model.feature_importances_
    }).sort_values(
        by="Importance",
        ascending=False
    )

    st.bar_chart(
        importance_df.set_index("Feature")
    )

    st.dataframe(
        importance_df,
        use_container_width=True
    )

    st.success(
        "Model evaluation completed."
    )


# =====================================================
# PREDICTION HISTORY
# =====================================================

elif page == "📜 Prediction History":

    st.title("📜 Prediction History")

    st.write(
        "View and download your previous flower predictions."
    )

    st.divider()

    if len(st.session_state.history) > 0:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Total Predictions",
            len(history_df)
        )

        col2.metric(
            "Species Predicted",
            history_df["Predicted Species"].nunique()
        )

        st.divider()

        st.dataframe(
            history_df,
            use_container_width=True
        )

        st.download_button(
            "⬇️ Download History CSV",
            data=history_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="iris_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button("🗑️ Clear Prediction History"):

            st.session_state.history = []

            st.rerun()

    else:

        st.info(
            "No prediction history available. "
            "Make a prediction to see it here."
        )


# =====================================================
# BULK PREDICTION
# =====================================================

elif page == "📁 Bulk Prediction":

    st.title("📁 Bulk Flower Prediction")

    st.write(
        "Upload a CSV file to predict multiple Iris flowers together."
    )

    st.divider()

    st.info(
        "Your CSV must contain these four columns:\n\n"
        + ", ".join(FEATURE_COLUMNS)
    )

    # SAMPLE CSV

    sample_data = pd.DataFrame(
        [
            [5.1, 3.5, 1.4, 0.2],
            [6.2, 3.4, 5.4, 2.3],
            [5.9, 3.0, 4.2, 1.5]
        ],
        columns=FEATURE_COLUMNS
    )

    st.download_button(
        "⬇️ Download Sample CSV",
        data=sample_data.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="iris_sample_data.csv",
        mime="text/csv"
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            data = pd.read_csv(uploaded_file)

            if data.empty:

                st.error("The uploaded CSV is empty.")

            else:

                missing_columns = [
                    col for col in FEATURE_COLUMNS
                    if col not in data.columns
                ]

                if missing_columns:

                    st.error(
                        "Missing columns: "
                        + ", ".join(missing_columns)
                    )

                else:

                    input_data = data[
                        FEATURE_COLUMNS
                    ].apply(
                        pd.to_numeric,
                        errors="coerce"
                    )

                    if input_data.isnull().any().any():

                        st.error(
                            "CSV contains missing or non-numeric values."
                        )

                    elif not np.isfinite(
                        input_data.to_numpy()
                    ).all():

                        st.error(
                            "CSV contains invalid numeric values."
                        )

                    else:

                        st.success(
                            f"{len(data)} records uploaded successfully."
                        )

                        st.subheader("📋 Uploaded Data")

                        st.dataframe(
                            data.head(10),
                            use_container_width=True
                        )

                        if st.button(
                            "🚀 Predict All Flowers",
                            type="primary",
                            use_container_width=True
                        ):

                            predictions = model.predict(
                                input_data
                            )

                            probabilities = model.predict_proba(
                                input_data
                            )

                            results = data.copy()

                            results["Predicted Species"] = predictions

                            results["Confidence (%)"] = (
                                np.max(
                                    probabilities,
                                    axis=1
                                ) * 100
                            ).round(2)

                            st.session_state.bulk_results = results

                            # SAVE BULK PREDICTIONS

                            for row_number, row in enumerate(
                                input_data.to_numpy()
                            ):

                                save_history(
                                    row.tolist(),
                                    predictions[row_number],
                                    np.max(probabilities[row_number]) * 100
                                )

                        if st.session_state.bulk_results is not None:

                            results = st.session_state.bulk_results

                            st.divider()

                            st.subheader("✅ Prediction Results")

                            st.metric(
                                "Flowers Processed",
                                len(results)
                            )

                            st.dataframe(
                                results,
                                use_container_width=True
                            )

                            st.download_button(
                                "⬇️ Download Prediction Results",
                                data=results.to_csv(
                                    index=False
                                ).encode("utf-8"),
                                file_name="iris_bulk_predictions.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

        except Exception as error:

            st.error(
                f"Unable to process CSV: {error}"
            )


# =====================================================
# ABOUT PROJECT
# =====================================================

elif page == "📘 About Project":

    st.title("📘 About Iris AI Classifier")

    st.write("""
    Iris AI Classifier is a Machine Learning project
    that predicts the species of an Iris flower.

    It uses four physical measurements to classify
    flowers into three species.

    The application uses a trained Random Forest
    classification model.
    """)

    st.divider()

    st.subheader("🎯 Project Objectives")

    st.write("""
    - Understand the Iris dataset.
    - Perform data visualization and analysis.
    - Train a Random Forest classification model.
    - Evaluate model performance.
    - Predict flower species through a web application.
    - Support multiple predictions using CSV files.
    """)

    st.divider()

    st.subheader("🛠️ Technologies Used")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.write("🐍 Python")

        st.write("📊 Pandas")

        st.write("🔢 NumPy")

    with col2:

        st.write("🤖 Scikit-learn")

        st.write("🌳 Random Forest")

        st.write("💾 Joblib")

    with col3:

        st.write("🖥️ Streamlit")

        st.write("📈 Matplotlib")

        st.write("📁 CSV Processing")

    st.divider()

    st.subheader("📐 Input Features")

    feature1, feature2 = st.columns(2)

    with feature1:

        st.write("1. Sepal Length (cm)")

        st.write("2. Sepal Width (cm)")

    with feature2:

        st.write("3. Petal Length (cm)")

        st.write("4. Petal Width (cm)")

    st.divider()

    st.subheader("🌺 Iris Species")

    for species in SPECIES_NAMES:

        st.write("🌸 " + species.capitalize())

    st.divider()

    st.subheader("🌳 Random Forest Algorithm")

    st.write("""
    Random Forest is an ensemble Machine Learning algorithm.

    It combines multiple decision trees and uses their
    predictions to make a final classification.

    It is used in this project to classify Iris flowers
    based on their measurements.
    """)

    st.success(
        "Project: Iris Flower Classification using Machine Learning"
    )

    st.caption(
        "Developed using Python, Scikit-learn and Streamlit."
    )