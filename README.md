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

![clipboard1](https://github.com/user-attachments/assets/a2dae1c4-5568-4fa3-8463-0a0e03aa6e47)

features = [module_temperature, irradiance, power = [current *voltage], panel_age, maintenance_count, soiling_ratio, error_code]



