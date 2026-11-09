# Student Mental Health Risk Prediction

## Overview

This project aims to analyze student mental health data and develop a machine learning system capable of predicting students' mental health risk levels (Low, Medium, or High).

The project combines data analysis, data preprocessing, machine learning, model evaluation, and workflow orchestration using Apache Airflow.

The main objective is to identify patterns and relationships between students' academic performance, lifestyle habits, social environment, and mental health risk.

**Note:** This project is intended for educational and research purposes only. Its predictions are not clinical diagnoses and should not replace professional mental health assessments.

## Dataset

The project uses the `student_mental_health.csv` dataset.

The dataset contains:
- 2,000 observations
- 17 features
- 3 target classes: Low, Medium, and High

### Features description

| Feature | Description |
|---|---|
| Student_ID | Unique student identifier |
| Age | Student's age |
| Gender | Gender category |
| Year_of_Study | Current year of study |
| Sleep_Hours | Average hours of sleep |
| Study_Hours_Per_Week | Weekly study hours |
| GPA | Grade Point Average |
| Social_Media_Hours | Hours spent on social media |
| Screen_Time_Hours | Daily screen time |
| Physical_Activity | Physical activity category |
| Financial_Stress | Financial stress level |
| Family_Support | Family support level |
| Attendance_Rate | Student attendance rate |
| Part_Time_Job | Whether the student has a part-time job |
| Diet_Quality | Diet quality category |
| Previous_Mental_Health_History | Previous mental health history |
| Mental_Health_Risk | Target variable |

### Target variable

The target variable is `Mental_Health_Risk`, which contains three possible classes:

- Low
- Medium
- High

Target distribution:
- High: 680 students
- Medium: 660 students
- Low: 660 students

## Exploratory Data Analysis (EDA)

The exploratory analysis is available in:

`2.exploratory_data_analysis.ipynb`

The notebook covers:

1. Dataset loading and initial inspection.
2. Dataset structure and statistical summaries.
3. Missing value analysis.
4. Duplicate detection.
5. Categorical variable validation.
6. Outlier detection using the IQR method.
7. Target variable distribution analysis.
8. Numerical feature distributions.
9. Relationships between features and mental health risk.
10. Spearman correlation analysis.

The analysis identified 419 missing cells and no duplicated rows.

## Project Workflow

The project is organized around the following machine learning workflow:

### 1. Data Collection and Loading
- Load the student mental health dataset.
- Inspect the dataset structure.
- Validate column names and data types.

### 2. Exploratory Data Analysis
- Analyze numerical and categorical features.
- Investigate missing values and outliers.
- Study relationships between variables.
- Analyze the distribution of the target variable.

### 3. Data Preprocessing
- Handle missing values.
- Treat outliers where appropriate.
- Encode categorical variables.
- Scale numerical variables.
- Handle class imbalance if necessary.

### 4. Model Training

The planned machine learning models include:

- Logistic Regression
- Random Forest
- Gradient Boosting

Models will be compared using consistent preprocessing pipelines.

### 5. Hyperparameter Optimization
- Apply cross-validation.
- Perform hyperparameter tuning.
- Select the best-performing model.

### 6. Model Evaluation

The evaluation process includes:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC where applicable

A stratified train/test split of 80/20 is planned.

### 7. Model Persistence
- Save the selected trained pipeline.
- Store model evaluation results.
- Implement reusable prediction functions.

### 8. Workflow Orchestration

Apache Airflow is used to orchestrate and automate the machine learning workflow.

### 9. Visualization and Deployment

A Streamlit interface is planned to:
- Display model performance metrics.
- Visualize prediction results.
- Support individual predictions.
- Support batch predictions.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.10 | Main programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning |
| Jupyter Notebook | Exploratory analysis |
| Apache Airflow 2.9.2 | Workflow orchestration |
| PostgreSQL 13 | Airflow metadata database |
| Docker | Containerization |
| Docker Compose | Service management |
| Streamlit | Planned user interface |

## Project Structure

```text
.
├── app/
│   └── Streamlit application
│
├── dags/
│   └── Airflow DAG definitions
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── Saved machine learning models
│
├── notebooks/
│   └── 2.exploratory_data_analysis.ipynb
│
├── reports/
│   └── Generated figures and evaluation reports
│
├── src/
│   └── Reusable Python modules
│
├── tests/
│   └── Automated tests
│
├── student_mental_health.csv
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Installation and Setup

### Prerequisites

Make sure you have installed:

- Python 3.10
- Docker
- Docker Compose
- Git

### 1. Clone the repository

```bash
git clone <repository-url>
cd <project-directory>
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Linux / macOS**
```bash
source .venv/bin/activate
```

**Windows**
```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the exploratory notebook

```bash
jupyter notebook 2.exploratory_data_analysis.ipynb
```

Make sure the dataset is available at the path expected by the notebook.

## Running with Docker

The project provides a Docker-based Apache Airflow environment with PostgreSQL.

### 1. Configure environment variables

Create a `.env` file in the project root:

```env
AIRFLOW_DB_USER=airflow
AIRFLOW_DB_PASSWORD=change_me
AIRFLOW_DB_NAME=airflow
AIRFLOW_FERNET_KEY=your_fernet_key
AIRFLOW_WEBSERVER_SECRET_KEY=your_secret_key
AIRFLOW_UID=50000
```

Use secure generated secrets and never commit the `.env` file.

### 2. Build and start the containers

```bash
docker compose up --build -d
```

### 3. Access Airflow

Open your browser and navigate to:

http://localhost:8080

### 4. Check container status

```bash
docker compose ps
```

### 5. View logs

```bash
docker compose logs -f airflow-webserver
```

```bash
docker compose logs -f airflow-scheduler
```

### 6. Stop the containers

```bash
docker compose down
```

To remove containers and the persisted PostgreSQL database volume:

```bash
docker compose down -v
```

**Warning:** The `-v` option deletes the persisted Airflow metadata database.

## Reproducibility and Best Practices

The project follows these intended practices:

- Use reusable Python functions for data processing.
- Separate training and testing datasets.
- Use stratified sampling.
- Avoid data leakage during preprocessing.
- Fit transformations only on training data.
- Apply cross-validation during model selection.
- Keep preprocessing and model training in a unified pipeline.
- Save the complete trained pipeline.
- Use reproducible random states.

## Limitations and Responsible Use

Mental health prediction is a sensitive application.

The project has several limitations:

- Dataset labels do not represent clinical diagnoses.
- Correlation does not imply causation.
- Missing values and dataset bias can affect model performance.
- Predictions may not generalize to all student populations.
- False negatives and false positives must be carefully considered.

The system should only be used for educational and research purposes, not for clinical decisions or decisions affecting students' rights and opportunities.

## Project Management

Project tasks and development activities are tracked using Jira.

The main project phases include:

1. Project setup and coding conventions.
2. Data loading and exploratory analysis.
3. Data preprocessing and transformation.
4. Model training and comparison.
5. Hyperparameter optimization and evaluation.
6. Airflow orchestration and Streamlit integration.
7. Final delivery and presentation.

## License

No license has yet been specified for this project. Add an appropriate license and dataset attribution before public distribution.

---

**Student Mental Health Risk Prediction**

*Data Science | Machine Learning | Python | Apache Airflow | Docker*
