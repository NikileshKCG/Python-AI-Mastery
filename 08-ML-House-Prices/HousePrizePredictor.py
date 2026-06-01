"""
Project 8: ML Model -- House Price Predictor
Concepts: scikit-learn pipeline, train/test split,
          feature engineering, multiple regression models,
          cross-validation, hyperparameter tuning,
          model evaluation metrics, feature importance,
          saving/loading models (joblib)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import joblib           # save and load training models
import os
warnings.filterwarnings("ignore")

# --- scikit-learn imports -----------------------------------
from sklearn.model_selection import (
    train_test_split,       # split data into train/test sets
    cross_val_score,        # k-fold cross validation
    GridSearchCV,           # hyperparameter tuning
)
from sklearn.preprocessing import (
    StandardScaler,         # normalize features (mean=0, std=1)
    LabelEncoder,           # encode categories as integers
    OneHotEncoder,          # encode categories as binary columns
    PolynomialFeatures,     # create interaction/polynomial features
)
from sklearn.pipeline import Pipeline   # chain preprocessing + model in one object
from sklearn.compose import ColumnTransformer 

# ------ Models -----------------------------------------------------
from sklearn.linear_model import (
    LinearRegression,           # baseline model
    Ridge,                      # L2 regularization
    Lasso,                      # L1 regularization (feature selction)
    ElasticNet,                 # L1 + L2 combined
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,          # bagging ensemble
    GradientBoostingRegressor,      # boosting ensemble
)
from sklearn.metrics import (
    mean_absolute_error,            # MAE
    mean_squared_error,             # MSE
    r2_score,                       # Rsquare - how much variance explained
)

#--------------------------------------------------------------
# SECTION 1: Generate Realistic Dataset
# Concepts: Synthetic data with real-world patterns
#           (used when you don't have real data yet)
#--------------------------------------------------------------

def generate_housing_data(n=1000):
    """
    Generate synthetic housing dataset with realistic patterns.
    Price depends on area, bedrooms, location, age, amenities.
    """
    np.random.seed(42)

    # Feature generation
    area_sqft       = np.random.randint(500, 5000, n)
    bedrooms        = np.random.randint(1, 7, n)
    bathrooms       = np.clip(bedrooms - np.random.randint(0, 2, n), 1, 5)
    house_age       = np.random.randint(0, 50, n)
    garage_cars     = np.random.randint(0, 4, n)
    floors          = np.random.randint(1, 4, n)
    garden          = np.random.randint(0, 2, n)        # 0/1 binary
    pool            = np.random.randint(0, 2, n)
    renovation      = np.random.randint(0, 2, n)

    neighborhoods   = np.random.choice(
        ["Downtown", "Suburb", "Rural", "Waterfront", "Industrial"],
        n, p=[0.25, 0.35, 0.20, 0.10, 0.10]     # weighted probabilities
    )
    condition       = np.random.choice(["Poor", "Fair", "Good", "Excellent"], n,
                                       p = [0.10, 0.25, 0.40, 0.25])
    
    # Neighborhood price multipliers
    neigh_mult      = {"Downtown": 1.5, "Suburb": 1.0, "Rural": 0.7,
                  "Waterfront": 2.0, "Industrial": 0.6}
    cond_mult       = {"Poor": 0.75, "Fair": 0.90, "Good": 1.0, "Excellent": 1.2}

    # Base price formula -- mimics real estate pricing
    base_price = (
        area_sqft * 120 +
        bedrooms * 8000 +
        bathrooms * 6000 +
        garage_cars * 5000 +
        garden * 10000 +
        pool * 25000 +
        renovation * 15000 +
        house_age * 500
    )

    # Apply multipliers
    price = np.array([
        base_price[i]
        * neigh_mult[neighborhoods[i]]
        * cond_mult[condition[i]]
        * np.random.uniform(0.88, 1.12)     # +or- 12% market noise
        for i in range(n)
    ]).astype(int)

    price = np.clip(price, 30000, 2000000)

    df = pd.DataFrame({
        "area_sqft"         : area_sqft,
        "bedrooms"          : bedrooms,
        "bathrooms"         : bathrooms,
        "house_age"         : house_age,
        "garage_cars"       : garage_cars,
        "floors"            : floors,
        "garden"            : garden,
        "pool"              : pool,
        "renovation"        : renovation,
        "neighborhood"      : neighborhoods,
        "condition"         : condition,
        "price"             : price,
    })

    return df 

#--------------------------------------------------------------
# SECTION 2: Exploratory Data Analysis (EDA)
# Concepts: Always explore before modeling
#--------------------------------------------------------------

def run_eda(df):
    print("\n ============ Exploratory Data Analysis ============= ")
    print(f"\nShape    : {df.shape}")
    print(f"Nulls      : {df.isnull().sum()}")
    print(f"\n Price Stats: ")
    print(f" Min        : ${df['price'].min():>12,}")
    print(f" Max        : ${df['price'].max():>12,}")
    print(f" Mean       : ${df['price'].mean():>12,.0f}")
    print(f" Median     : ${df['price'].median():>12,.0f}")
    print(f" Std        : ${df['price'].std():>12,.0f}")

    print(f" \nNeighborhood Distribution.")
    print(df["neighborhood"].value_counts().to_string())

    print(f"\nCorrelation with Price. ")
    numeric = df.select_dtypes(include=np.number)
    corr    = numeric.corr()["price"].drop("price").sort_values(ascending=False)
    for col, val in corr.items():
        bar = " " * int(abs(val) * 30)
        sign = "+" if val > 0 else "-"
        print(f" {col:<15} {sign}{abs(val):.3f} {bar}")
    
    # EDA Visualization
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(" EDA - House Price Dataset", fontsize=14, color="white")
    fig.patch.set_facecolor("#0f0f0f")
    for ax in axes.flat:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#aaa")
        ax.title.set_color("white")
        ax.xaxis.label.set_color("#aaa")
        ax.yaxis.label.set_color("#aaa")

    # Price Distribution
    axes[0, 0].hist(df["price"]/1000, bins=40, color="#e94560", edgecolor="#0f0f0f", alpha=0.8)
    axes[0, 0].set_title(" Price Distribution ")
    axes[0, 0].set_xlabel("Price ($K)")

    # Area vs Price Scatter
    axes[0, 1].scatter(df["area_sqft"], df["price"]/1000, alpha=0.3, color="#533483", s=15)
    axes[0, 1].set_title(" Area Vs Price ")
    axes[0, 1].set_xlabel("Area (sqft)"); axes[0, 1].set_ylabel("Price ($K)")

    # Avg price by neighborhood
    neigh_price = df.groupby("neighborhood")["price"].mean().sort_values() / 1000
    axes[0, 2].barh(neigh_price.index, neigh_price.values,
                    color=["#e94560", "#0f3460", "#533483", "#05c46b", "#ffd460"])
    axes[0, 2].set_title(" Average Price by Neighborhood ")
    axes[0, 2].set_xlabel("Price ($K)")

    # Boxplot by condition 
    df.boxplot(column="price", by="condition", ax=axes[1, 0],
               patch_artist=True)
    axes[1, 0].set_title(" Price by Condition ")
    axes[1, 0].set_xlabel("Condition")
    plt.sca(axes[1, 0]); plt.title("Price by Condition")

    # Bedrooms vs price
    bed_price = df.groupby("bedrooms")["price"].mean() / 1000
    axes[1, 1].bar(bed_price.index, bed_price.values, color="#05c46b", edgecolor="#0f0f0f", alpha=0.8)
    axes[1, 1].set_title(" Average Price by Bedrooms ")
    axes[1, 1].set_xlabel("Bedrooms")
    axes[1, 1].set_ylabel("Average Price ($K)")

    # Correlation heatmap
    numeric_cols = ["area_sqft", "bedrooms", "bathrooms", "house_age", "garage_cars", "price"]
    corr_matrix  = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, ax=axes[1, 2], annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5,
                annot_kws={"size":7}, cbar=False)
    axes[1, 2].set_title(" Correlation Matrix ")

    plt.tight_layout()
    plt.savefig("eda_plots.png", dpi=130, bbox_inches="tight", facecolor="#0f0f0f")
    print("\n EDA plots saved: eda_plots.png")
    plt.show()
    plt.close()

#--------------------------------------------------------------
# SECTION 3: Feature Engineerig
# Concepts: Creating new features form existing ones
#           This is where domain knowledge meets ML
#--------------------------------------------------------------

def engineer_features(df):
    """ Add derived features that help the model learn better. """
    df = df.copy()

    # Ration features -- often more informative than raw values
    df["price_per_sqft"]    = df["price"] / df["area_sqft"]
    df["bath_bed_ratio"]    = df["bathrooms"] / df["bedrooms"]
    df["area_per_bedroom"]  = df["area_sqft"] / df["bedrooms"]
    df["total_rooms"]       = df["bedrooms"] + df["bathrooms"]

    # Binary feature: is it a new house ?
    df["is_new"]            = (df["house_age"] <= 5).astype(int)

    # Luxury score - combination of premium features
    df["luxury_score"]      = (
        df["pool"] * 3 +
        df["garden"] * 2 +
        df["garage_cars"] +
        df["renovation"] * 2
    )

    # Log transform target -- price is right-skewed, log makes it normal
    # This is a CRITICAL preprocessing step for regression
    df["log_price"] = np.log1p(df["price"])

    print(f" Feature Engineering done. New shape: {df.shape}")
    return df

#--------------------------------------------------------------
# SECTION 4: Preprocessing Pipeline
# Concepts: sklearn Pipeline chains steps so you
#           never accidentally apply scaling to test data
#           before fitting on train data (data leakage!)
#--------------------------------------------------------------

def build_preprocessor(numeric_features, categorical_features):
    """
    Build a ColumnTransformerr that:
    - Scales numeric columns with StandardScaler
    - One-hot encodes categorical columns
    """
    numeric_transformer     = Pipeline([("scalar", StandardScaler())])
    categorical_transformer = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore",
                                                                 sparse_output=False))])
    
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
         ("cat", categorical_transformer, categorical_features),
    ])
    return preprocessor

#--------------------------------------------------------------
# SECTION 5: Train & Evaluate Multiple Models
# Concepts: Model comparison, evaluation metrics, 
#           cross-validation to detect overfitting 
#--------------------------------------------------------------

def evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    """ Train and compare 6 regression models. """

    models = {
        "Linear Regression" : LinearRegression(),
        "Ridge"             : Ridge(alpha=10),
        "Lasso"             : Lasso(alpha=100),
        "Decision Tree"     : DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest"     : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting" : GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    
    results = []
    trained_models = {}

    print(f"\n {'Model':<22} {'MAE':>10} {'RMSE':>10} {'R':>8} {'CV R':>10}")
    print(" " + "--" * 65)

    for name, model in models.items():
        # Full pipeline: preprocessing + model
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # Train
        pipe.fit(X_train, y_train)

        # Predict
        y_pred = pipe.predict(X_test)

        # Metrics
        mae     = mean_absolute_error(y_test, y_pred)
        rmse    = np.sqrt(mean_squared_error(y_test, y_pred))
        r2      = r2_score(y_test, y_pred)

        # Cross-validation -- train on 5 different splits, average R
        # This catches overfitting: if CV score << test score, model memorized training data
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
        cv_r2     = cv_scores.mean()

        print(f" {name:<22} ${name:>9,.0f} ${rmse:>9,.0f} {r2:>8.4f} {cv_r2:>10.4f}")

        results.append({
            "model": name, "MAE": mae, "RMSE": rmse,
            "R2": r2, "CV_R2": cv_r2
        })
        trained_models[name] = pipe

    return pd.DataFrame(results), trained_models

#--------------------------------------------------------------
# SECTION 6: Hyperparameter Tuning
# Concepts: GridSearchCV -- try every combination
#           of hyperparameters, pick the best one
#--------------------------------------------------------------

def tune_random_forest(X_train, y_train, preprocessor):
    """ Tune Random Forest with GridSearchCV."""
    print("\n Tuning Random Forest (this takes -30 seconds)... ")

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    # Parameter grid -- every combination will be tried 
    param_grid = {
        "model__n_estimators"   : [50, 100, 200],      # number of trees
        "model__n_max_depth"    : [None, 10, 20],       # max tree depth
        "model__n_samples_split": [2, 5],               # min samples to split a mode
    }

    # GridSearchCV with 3-fold CV
    grid_search = GridSearchCV(
        pipe, param_grid,
        cv=3, scoring="r2",
        n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)

    print(f" Best params : {grid_search.best_params_}")
    print(f" Best CV R2 : {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

#-----------------------------------------------------------------
# SECTION 7: Feature Importance
# Concepts: Which features actually drive predictions?
#           Critical for understanding and explaining your model
#-----------------------------------------------------------------

def plot_feature_importance(best_model, feature_names):
    """ Plot feature importances from the best tree-based model."""
    try:
        rf_model = best_model.named_steps["model"]
        importances = rf_model.feature_importances_

        # Get feature names after one-hot encoding
        preprocessor = best_model.named_steps["model"]
        cat_features = preprocessor.named_transformers_["cat"] \
                        .named_steps["onehot"].get_feature_names_out()
        num_features = preprocessor.transformers_[0][2]
        all_features = list(num_features) + list(cat_features)

        # Create Series and sort
        importance_series = pd.Series(importances, index=all_features) \
                                .sort_values(ascending=True).tail(15)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor("#0f0f0f")
        ax.set_facecolor("#1a1a2e")

        colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(importance_series)))
        bars   = ax.barh(importance_series.index, importance_series.values,
                         color = colors, edgecolor="#0f0f0f")
        
        ax.set_title(" Top 15 Feature Importance (Random Forest)",
                     color="white", fontsize=13)
        ax.set_xlabel("Importance", color="#aaa")
        ax.tick_params(colors="#aaa")
        ax.grid(axis="X", alpha=0.3, color="444")

        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                     f"{width:.3f}", va="center", color="white", fontsize=8)
        
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=130, bbox_inches="tight",
                    facecolor="#0f0f0f")
        print(" Saved: feature_importance.png")
        plt.show()
        plt.close()

    except Exception as e:
        print(f" Could not plot feature importance: {e}")


#-----------------------------------------------------------------------
# SECTION 8: Residual Analysis
# Concepts: Check model errors -- are they random?
#           If patterns exist in residuals - model is missing something
#-----------------------------------------------------------------------

def plot_residuals(best_model, X_test, y_test):
    """ Actual vs Predicted + Residual plots. """
    y_pred      = best_model.predict(X_test)
    residuals   = y_test - y_pred

    fig, axes   = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Model Evaluation -- Residual Analysis", fontsize=13, color="white")
    fig.patch.set_facecolor("#0f0f0f")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#aaa")
        ax.title.set_color("white")
        ax.xaxis.label.set_color("#aaa")
        ax.yaxis.label.set_color("#aaa")
    
    # Actual vs Predicted
    max_val = max(y_test.max(), y_pred.max())
    axes[0].scatter(y_test/1000, y_pred/1000, alpha=0.4, color="#533483", s=15)
    axes[0].plot([0, max_val/1000], [0, max_val/1000],
                 color="#e94560", linewidth=2, linestyle="--", label="Perfect fit")
    axes[0].set_title(" Actual vs Predicted")
    axes[0].set_xlabel(" Actual Price ($K)")
    axes[0].set_ylabel(" Predicted Price ($K)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    # Residuals vs Predicted
    axes[1].scatter(y_pred/1000, residuals/1000, alpha=0.4, color="#05c46b", s=15)
    axes[1].axhline(0, color="#e94560", linewidth=1.5, linestyle="--")
    axes[1].set_title(" Residuals vs Predicted ")
    axes[1].set_xlabel(" Predicted Price ($K))")
    axes[1].set_ylabel(" Residuals ($K)")
    axes[1].grid(alpha=0.2)

    # Residual distribution
    axes[2].hist(residuals/1000, bins=35, color="#ffd460", edgecolor="#0f0f0f", alpha=0.8)
    axes[2].axvline(0, color="#e94560", linewidth=1.5, linestyle="--")
    axes[2].set_title(" Residuals Distribution ")
    axes[2].set_xlabel(" Residuals ($K))")
    axes[2].grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig("residual_analysis.png", dpi=130, bbox_inches="tight", facecolor="#0f0f0f")
    print(" Saved: residual_analysis.png")
    plt.show(); plt.close()

#--------------------------------------------------------------
# SECTION 9: Save & Load Model
# Concepts: joblib serializes trained model to disk
#           Deploy it anywhere without retraining
#--------------------------------------------------------------

def save_model(model, filename="house_price_model.pk1"):
    joblib.dump(model, filename)
    size = os.path.getsize(filename) / 1024
    print(f" Model saved: {filename} ({size:.1f} KB)")

def load_model(filename="house_price_model.pk1"):
    if not os.path.exists(filename):
        print(f" Model file not found : {filename}")
        return None
    model = joblib.load(filename)
    print(f" Model loaded: {filename}")
    return model

#--------------------------------------------------------------
# SECTION 10: Predict on New House
# Concepts: Using the trained pipeline to predict 
#           on brand-new unseen data
#--------------------------------------------------------------

def predict_new_house(model):
    """ Interactively predict price for a custom house. """
    print("\n Enter house details to predict price: \n")

    try:
        area              = float(input("Area (sqft)        : "))
        bedrooms          = int(input(" Bedrooms            : "))
        bathrooms         = int(input(" Bathrooms           : "))
        house_age         = int(input(" House age (years)   : "))
        garage_cars       = int(input(" Garage cars (0-3)   : "))
        floors            = int(input(" Floors              : "))
        garden            = int(input(" Garden? (1/0)       : "))
        pool              = int(input(" Pool? (1/0)         : "))
        renovation        = int(input(" Renovated? (1/0)    : "))
        print(" Neighborhood options: Downtown / Suburb / Rural / Waterfront / Industrial")
        neighborhood      = (input(" Neighborhood           : ")).strip().title()
        print(" Condition options: Poor / Fair / Good / Excellent")
        condition         = (input(" Condition              : ")).strip().title()

        # Build DataFrame exactly as traning data
        house = pd.DataFrame([{
        "area_sqft"         : area,
        "bedrooms"          : bedrooms,
        "bathrooms"         : bathrooms,
        "house_age"         : house_age,
        "garage_cars"       : garage_cars,
        "floors"            : floors,
        "garden"            : garden,
        "pool"              : pool,
        "renovation"        : renovation,
        "neighborhood"      : neighborhood,
        "condition"         : condition,
        }])
    
        # Apply same feature engineering as training
        house["bath_bed_ratio"]     = house["bathrooms"] / house["bedrooms"]
        house["area_per_dedroom"]   = house["area_sqft"] / house["bedrooms"]
        house["total_rooms"]        = house["bedrooms"] + house["bathrooms"]
        house["is_new"]             = (house["house_age"] <= 5).astype(int)
        house["luxury_score"]       = (
        house["pool"] * 3 + house["garden"] * 2 +
        house["garage_cars"] + house["renovation"] * 2
        )

        predicted_price = model.predict(house)[0]

        print(f"""
        =============================================================
        ||             PRICE PREDCITION                             ||
        ||===========================================================
        ||   Predicted Pirce : ${predicted_price:>14,.0f}           ||
        ||   Per sqft        : ${predicted_price/area:>14,.0f}      ||
        =============================================================""")

    except Exception as e:
        print(f" Error: {e}")
    
#--------------------------------------------------------------
# SECTION 11: Full ML pipeline Runner
#--------------------------------------------------------------

def run_full_pipeline():
    """ Execute the complete ML workflow end to end. """
    print("\n Running Full ML Pipeline... \n")

    # Step 1 : Data
    print(" [1/7] Genrating Dataset... \n")
    df = generate_housing_data(1000)

    # Step 2 : EDA
    print(" [2/7] Exploratory Data Analysis... \n")
    run_eda(df)

    # Step 3 : Feature Engineering
    print(" [3/7] Feature Engineering ... \n")
    df = engineer_features(df)

    # Step 4 : Split
    print(" [4/7] Train/Test Split (80/20)... \n")
    feature_cols = [
        "area_sqft", "bedrooms", "bathrooms", "house_age",
        "garage_cars", "floors", "garden", "pool", "renovation",
        "bath_bed_ratio", "area_per_bedroom", "total_rooms",
        "is_new", "luxury_score",
        "neighborhood", "condition"    
    ]
    numeric_features    = [c for c in feature_cols if df[c].dtype != "object"]
    categorical_features = [c for c in feature_cols if df[c].dtype == "object"]

    X = df[feature_cols]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f" Traing: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    # Step 5 : Preprocessing + Model Comparison
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    print(" [5/7] Training and Comparing Models ... ")
    results_df, trained_models =evaluate_models(
        X_train, X_test, y_train, y_test, preprocessor
    )

    best_name = results_df.sort_values("R2", ascending=False).iloc[0]["model"]
    print(f"\n Best Model: {best_name}")

    # Step 6 : Hyperparameter Tuning
    print(" [6/7] Hyperparameter Tuning (Random Forest) ... ")
    preprocessor2 = build_preprocessor(numeric_features, categorial_features)
    best_model = tune_random_forest(X_train, y_train, preprocessor2)

    # Final evaluation
    y_pred      = best_model.predict(X_test)
    final_r2    = r2_score(y_test, y_pred)
    final_mae   = mean_absolute_error(y_test, y_pred)
    final_rmse  = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n -------- Tuned Model Final Metrics --------")
    print(f" R2     : {final_r2} (1.0 = perfect)")
    print(f" MAE    : ${final_mae:,.0f} (avg prediction error)")
    print(f" RMSE   : ${final_rmse:,.0f} (penalizes large errors)")

    # Step 7 : Plots + Save
    print(" [7/7] Genearting plots & Saving model ... ")
    plot_feature_importance(best_model, feature_cols)
    plot_residuals(best_model, X_test, y_test)
    save_model(best_name)

    print("\n Full pipeline complete!")
    return best_model, X_test, y_test, numeric_features, categorical_features

#----------------------------------------------------------
# SECTION 12: Main Menu
#-----------------------------------------------------------

trained_model = None

def main():
    global trained_model

    print("\n" + "="* 52)
    print(" HOUSE PRICE PREDICTION - ML MODEL ")
    print("="*52)

    while True:
        print("""
---------------- MAIN MENU ------------------------------------
              [1]  Run Full ML Pipeline (recommended first)
              [2]  EDA only
              [3]  Load Saved Model
              [4]  Predict Price for New House
              [0]  Exit
-------------------------------------------------------------""")
        
        choice = input(" Enter Choice: ").strip()

        try:
            if choice == "1":
                trained_model, *_ = run_full_pipeline()

            elif choice == "2":
                df = generate_housing_data(1000)
                run_eda(df)

            elif choice == "3":
                trained_model = load_model()

            elif choice == "4":
                if trained_model is None:
                    print(" Train or load a model first (option 1 or 3).")
                else:
                    predict_new_house(trained_model)

            elif choice == "0":
                print("\n GoodBye!!!\m")
                break

            else:
                print(" Invalid Choice, try again!")
        
        except Exception as e:
            print(f" Error: {e}")
            import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()