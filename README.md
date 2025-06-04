# zelestra_challenge
 Zelestra X AWS ML Ascend Challenge  
 
 First, we need to clean the data. We should use every row to make the model more accurate. Need to figure out ways to handle missing data and create new columns from the ones we already have. We can split the columns among ourselves to clean things up faster.  
 
``` mermaid
graph TD
    A[Training Data 20000 rows] --> B[Data Preprocessing]
    B --> C[Missing Values]
    B --> D[Feature Engineering]
    B --> E[Categorical Encoding]
    
    C --> C1[Replace zeros with NaN]
    C --> C2[Fill with mean or median]
    
    D --> D1[Create power feature]
    D --> D2[Create temperature difference]
    D --> D3[Create effective irradiance]
    D --> D4[Create age degradation factor]
    
    E --> E1[Encode string ID]
    E --> E2[Encode error codes]
    E --> E3[Encode installation types]
    
    C2 --> F[Clean Dataset]
    D4 --> F
    E3 --> F
    
    F --> G[Train Validation Split]
    
    G --> H1[Random Forest Model]
    G --> H2[Gradient Boosting Model]
    G --> H3[Ridge Regression Model]
    G --> H4[XGBoost Model]
    
    H1 --> I1[Prediction A]
    H2 --> I2[Prediction B]
    H3 --> I3[Prediction C]
    H4 --> I4[Prediction D]
    
    I1 --> J[Ensemble Combination]
    I2 --> J
    I3 --> J
    I4 --> J
    
    J --> K1[Simple Average Method]
    J --> K2[Weighted Average Method]
    J --> K3[Stacking Method]
    
    K1 --> L[Final Ensemble Prediction]
    K2 --> L
    K3 --> L
    
    L --> M[Test Data Processing]
    M --> N[Apply Same Preprocessing]
    N --> O[Apply All Models]
    O --> P[Apply Ensemble Weights]
    P --> Q[Final Submission File]
    
    subgraph Val [Validation Process]
        R[Cross Validation]
        S[Calculate RMSE]
        T[Calculate Final Score]
    end
    
    G -.-> R
    L -.-> S
    
    subgraph Benefits [Key Benefits]
        U[Reduces Overfitting]
        V[Better Generalization]
        W[Higher Accuracy]
        X[Robust Predictions]
    end
```


## Data Preprocessing

### Data Cleaning

1. **Handling Missing Values**:
   - **Panel Age**: Imputed missing values using the median age of panels grouped by `string_id`.
   - **Maintenance Count**: Used a hierarchical approach to impute missing values based on `string_id` and `error_code`.
   - **Soiling Ratio**: Imputed using median values based on maintenance count bins.
   - **Module Temperature**: Applied linear regression and KNN imputation to fill missing values.
   - **Power**: Created a robust power feature using current and voltage, handling missing values with KNN imputation.
   - **Irradiance**: Imputed missing values using physical relationships and KNN imputation.
   - **Humidity**: Imputed missing values using KNN imputation.
   - **Error Code**: Filled missing values with a new category 'Unknown'.

2. **Outlier Removal**:
   - Used the Interquartile Range (IQR) method to detect and remove outliers from numerical features.

### Feature Engineering

- **Power Feature**: Created a new feature representing the power output of the panels using current and voltage.
- **Encoding Categorical Variables**: One-hot encoded the `error_code` categorical variable to prepare for modeling.

### Data Visualization

- **Distribution Plots**: Visualized the distribution of numerical features to understand their spread and identify any anomalies.
- **Correlation Matrix**: Analyzed the correlation between features to understand relationships and guide feature selection for modeling.

### Data Preparation for Modeling

- **Feature Scaling**: Standardized numerical features to ensure they are on a similar scale, which is crucial for many machine learning algorithms.
- **Saving Cleaned Data**: The cleaned and preprocessed data is saved to a CSV file for use in subsequent modeling steps.

## Repository Structure

```
zelestra_challenge/
├── data_preprocessing/     # Scripts and notebooks for data preprocessing
├── dataset/               # Raw dataset files
├── processed_dataset/     # Cleaned and processed datasets
├── scaler/               # Saved scaler objects for data normalization
├── venv/                 # Python virtual environment
├── training_model.ipynb  # Main notebook for model training
├── requirements.txt      # Python package dependencies
└── README.md            # Project documentation
```

## Setup and Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the data preprocessing scripts in the `data_preprocessing` directory
2. Open and run the `training_model.ipynb` notebook to train and evaluate the models




