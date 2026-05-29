"""
Project 6:  Data Analysis with Pandas & NumPy
Concepts: numpy arrays, pandas Series & DataFrame,
          loading data, cleaning, filtering, groupby,
          merging, pivot tables, aggregation, export
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os
import warnings
warnings.filterwarnings("ignore")

#--------------------------------------------------------------
# SECTION 1: Numoy Fundamentals
# Concepts: ndarray, vectorized ops, boradcasting,
#           math functions -- the engine under pandas
#--------------------------------------------------------------

def numpy_basics():
    print("\n =============== NumPy Basics =============== ")

    # Creating arrays -- far faster than Python lists for math
    arr1  = np.array([10, 20, 30, 40, 50])
    arr2  = np.arrange(1, 6)            # [1, 2, 3, 4, 5]
    arr3  = np.linspace(0, 1, 5)        # 5 evenly space between 0 and 1
    zeros = np.zeros((3, 3))            # 3x3 matrix of zeros
    ones  = np.ones((2, 4))             # 2x4 matrix of ones
    rand  = np.random.rand(3, 3)        # random floats 0-1

    print(f" \n arr1        : {arr1}")
    print(f" arrange        : {arr2}")
    print(f" linspace       : {arr3}")
    print(f" zeros (3X3)    :\n{zeros}")

    # Vectorized operations -- applied to EVERY element at once (no loop needed)
    print(f"\n arr1 * 2     : {arr1 * 2}")
    print(f" arr1 + arr2    : {arr1 + arr2}")   #element-wise addition
    print(f" arr1 ** 2      : {arr1 ** 2}")
    print(f" sqrt(arr1)     : {np.sqrt(arr1)}")

    # Statistics
    data = np.random.randint(50, 100, size=20)
    print(f"\n Random data  : {data}")
    print(f" Mean           : {np.mean(data):.2f}")
    print(f" Std Dev        : {np.std(data):.2f}")
    print(f" Min / Max      : {np.min(data) / {np.max(data)}}")
    print(f" Median         : {np.median(data):.2f}")
    print(f" 25th pctile    : {np.percentile(data, 25):.2f}")

    # Slicing & Indexing (same syntax as Python lists but N-dimensional)
    matrix = np.array([[1,2,3], [4,5,6], [7,8,9]])
    print(f"\n Matrix   :\n{matrix}")
    print(f" Row 0      : {matrix[0]}")
    print(f" Column 1   : {matrix[:, 1]}")      # all rows, column index 1
    print(f" Submatrix  : \n{matrix[0:2, 1:3]}")

    # Boolean masking -- filter by condition
    print(f"\n data > 80    : {data[data > 80]}")   #only values > 80
    print(f" Count > 80     : {np.sum(data > 80)}") # True=1, sum counts them

#--------------------------------------------------------------
# SECTION 2: Generate Sample Dataset
# Concepts: Building a realistic DataFrame from scratch
#--------------------------------------------------------------

def generate_sales_dataset(n=200):
    """
    Generate a synthetic sales dataset.
    Returns a pandas DataFrame -- the core data structure.
    """
    random.seed(42)
    np.random.seed(42)

    regions     = ["North", "South", "East", "West"]
    categories  = ["Electronics", "Clothing", "Food", "Books", "Sports"]
    products    = {
        "Electronis": ["Laptop", "Phone", "Table", "Headphones"],
        "Clothing"  : ["Shirts", "Jeans", "Jacket", "Shoes"],
        "Food"      : ["Croceries", "Snacks", "Beverages", "Frozen"],
        "Books"     : ["Fiction", "Non-Fiction", "Textbook", "Comics", "Novels"],
        "Sports"    : ["Football", "Cricket Kit", "Gym Equipment", "Yoga Mat"]
    }
    salespeople = ["Ali", "Sara", "Raza", "Ayesha", "Bilal", "Hina", "Omar", "Zara"]

    start_date = datetime(2027, 1, 1)
    rows = []

    for i in range(n):
        category    = random.choice(categories)
        product     = random.choice(products[category])
        region      = random.choice(regions)
        salesperson = random.choice(salespeople)
        units       = random.randint(1, 50)
        unit_price  = round(random.uniform(10, 500), 2)
        discount    = round(random.uniform(0, 0.3), 2)  # 0-30% discount
        revenue     = round(units * unit_price * (1 - discount), 2)
        date        = start_date + timedelta(days=random.randint(0, 364))
        returned    = random.random() < 0.05

        rows.append({
            "order_id"      : f"ORD{1000 + i}",
            "date"          : date.strftime("%Y-%m-%D"),
            "region"        : region,
            "salesperson"   : salesperson,
            "category"      : category,
            "product"       : product,
            "units"         : units,
            "unit_price"    : unit_price,
            "discount"      : discount,
            "revenue"       : revenue,
            "returned"      : returned,
        })
    
    df = pd.DataFrame(rows)             # list of dicts - DataFrame
    return df

#--------------------------------------------------------------
# SECTION 3: DataFrame Basics
# Concepts: Series, DataFrame, dtypes, info, describe
#--------------------------------------------------------------

def dataframe_basics(df):
    print("\n ================ DataFrame Basics ================= ")

    print(f"\n Shape    : {df.shape}")      # (rows, columns)
    print(f" Columns    : {list(df.columns)}")
    print(f"\n First 3 rows: ")
    print(df.head(3).to_string(index=False))

    print(f"\n Data Types: ")
    print(df.dtypes)

    print(f"\n Statistical Summary: ")
    print(df.describe().round(2))

    print(f"\n Null values per column:")
    print(df.isnull().sum())

#--------------------------------------------------------------
# SECTION 4: Data Cleaning
# Concepts: handling nulls, duplicates, type casting,
#           string operations on columns
#--------------------------------------------------------------

def clean_data(df):
    print("\n ================= DATA CLEANING ================= ")

    # Inject some mess to clean
    df_dirty = df.copy()
    df_dirty.loc[0:4, "revenge"] = None         # add nulls
    df_dirty.loc[5:7, "region"] = " north "     # add whitespace/case issues
    df_dirty = pd.concat([df_dirty, df_dirty.iloc[:3]], ignore_index=True) # add duplicates

    print(f" Before Cleaning : {df_dirty.shape}")
    print(f" Null Count      : {df_dirty['revenue'].isnull().sum()}")
    print(f" Duplicates      : {df_dirty.duplicated().sum()}")

    # Fix nulls -- fill with column median 
    df_dirty["revenue"] = df_dirty["revenue"].fillna(df_dirty["revenue"].median())

    # Fix string inconsistencies -- strip whitespace, title case
    df_dirty["region"] = df_dirty["region"].str.strip().str.title()

    # Drop duplicates
    df_dirty = df_dirty.drop_duplicates()

    # Convert date column to datetime type
    df_dirty["month"]      = df_dirty["date"].dt.month
    df_dirty["month_name"] = df_dirty["date"].dt.strftime("%B")
    df_dirty["quarter"]    = df_dirty["date"].dt.quarter
    df_dirty["weekday"]    = df_dirty["date"].dt.day_name()

    print(f"\n After cleaning: {df_dirty.shape}")
    print(f" Null count: {df_dirty['revenue'].isnull().sum()}")
    print(f" Duplicates: {df_dirty.duplicated().sum()}")

    return df_dirty

#--------------------------------------------------------------
# SECTION 5: Filtering & Selection
# Concepts: boolean indexing, .loc, ,iloc, .query()
#--------------------------------------------------------------

def filtering_demo(df):
    print("\n ================== Filtering & Selection ================== ")

    # Boolean indexing -- same idea as numpy masking
    high_revenue = df[df["revenue"] > 5000]
    print(f"\n Orders > $5000 revenue : {len(high_revenue)}")

    # Multiple conditions - & (and), | (or), -(not)
    electronics_north = df[(df["category"] == "Electronics") & (df["region"] == "North")]
    print(f" Electronics in North : {len(electronics_north)}")

    # .query() -- SQL - like syntax, very readable
    top_orders = df.query("revenue > 3000 and discount < 0.1")
    print(f" High revenue, low disc : {len(top_orders)}")

    # .Ioc -- label-based : df.loc[row_label, col_label]
    print(f"\n First row (loc)  :")
    print(df.loc[0, ["order_id", "product", "revenue"]])

    # .iloc -- position-based df.iloc[rwo_idx, col_idx]
    print(f"\n Top-left 3x3 (iloc)  :")
    print(df.iloc[0:3, 0:3])

    # .isin() -- filter by list of values
    selected_regions = df[df["region"].isin(["North", "South"])]
    print(f"\n North + South Orders : {len(selected_regions)}")

    # .str accessor -- string operations on text columns
    shirt_orders = df[df["product"].str.contains("Phone|Laptop", case=False, na=False)]
    print(f" Phone or Laptop orders : {len(shirt_orders)}")

#------------------------------------------------------------------------------
# SECTION 6: GroupBy & Aggregation
# Concepts: split-apply-combine, agg(), transform()
#    This is the pandas superpower - used constantly in ML feature engineering
#------------------------------------------------------------------------------

def groupby_analysis(df):
    print("\n ================ GroupBy & Aggregation ================")

    # Simple groupby -- total revenue per region
    region_revenue = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    print(f"\n Revenue by Region: ")
    for region, rev in region_revenue.items():
        print(f" {region:<10} ${rev:>10,.2f}")
    
    # Multiple aggregations at once with .agg()
    category_stats = df.groupby("category").agg(
        total_orders  = ("order_id", "count"),
        total_revenue = ("revenue", "sum"),
        avg_revenue   = ("revenue", "mean"),
        avg_discount  = ("discount", "mean"),
    ).round(2).sort_values("total_revenue", ascending=False)

    print(f"\n Category Performance.")
    print(category_stats.to_string())

    #Groupby multiple columns
    region_category = df.groupby(["region", "category"])["revenue"].sum().unstack()
    print(f"\n Revenue by Region x Category: ")
    print(region_category.round(2).to_string())

    # Top salesperson per region - groupby + idxmax
    top_sales       = df.groupby(["region", "salesperson"])["revenue"].sum()
    top_per_region  = top_sales.groupby(level=0).idxmax()
    print(f"\n Top Salesperson per Region: ")
    for region, (reg, person) in top_per_region.items():
        rev = top_sales[(reg, person)]
        print(f"  {region:<8} -> {person:<10} ${rev:,.2f}")
    
    # transform() --> add group-level stat as a new column (for ML feature engineering)
    df["region_avg_revenue"] = df.groupby("region")["revenue"].transform("mean")
    df["above_region_avg"] = df["revenue"] > df["region_avg_revenue"]
    print(f"\n Orders above their region avg: {df['above_region_avg'].sum()}")

    return df

#--------------------------------------------------------------
# SECTION 7: Pivot Tables
# Concepts: Excel-style pivot tables in one line
#--------------------------------------------------------------

def pivot_analysis(df):
    print("\n ================= Pivot Tables =================")

    # Revenue pivot: rows=category, cols=region, values=revenue
    pivot = pd.pivot_table(
        df,
        values      = "revenue",
        index       = "category",
        columns     = "region",
        aggfunc     = "sum",
        fill_value  = 0,
        margins     = True, # add row/col totals
        margins_name= "TOTAL"
    ).round(2)

    print(f"\n Revenue Pivot (Category x Region):")
    print(pivot.to_string())

    # Units sold pivot with count
    units_pivot = pd.pivot_table(
        df,
        values      = "units",
        index       =  "category",
        columns     = "region",
        aggfunc     = "sum",
        fill_value  = 0
    )
    print(f"\n Units Sold Pivot: ")
    print(units_pivot.to_string())

#--------------------------------------------------------------
# SECTION 8: Merging & Joing DataFrames
# Concepts: merge (SQL-style joins), concat
#--------------------------------------------------------------

def merge_demo(df):
    print("\n ============== Merging DataFrames ===============")

    # Create a salesperson targets table
    targets = pd.DataFrame({
        "salesperson" : ["Raj", "Sara", " Mike", "Shyam", "Rani", "Uttam", "Mounica"],
        "target"      : [15000, 18000, 12000, 20000, 40000, 16000, 1000000],
        "team"        : ["A", "B", "C", "A", "C", "B", "A+"]  
    })

    # Aggregate actual revenue per salesperson 
    actual          = df.groupby("salesperson")["revenue"].sum().reset_index()
    actual.columns  = ["salesperson", "actual_revenue"]

    # INNER JOIN -> only matching rows (like SQL INNER JOIN)
    merged = pd.merged(actual, targets, on="salesperson", how="inner")

    merged["achievement_%"] = (merged["actual_revenue"] / merged["target"] * 100).round(1)
    merged["gap"]           = (merged["target"] - merged["actual_revenue"]).round(2)
    merged = merged.sort_values("achievement_%", ascending=False)

    print(f"\n salesperson Performance vs Target: ")
    print(merged[["salesperson", "team", "actual_revenue", "target", "achievement_%"]].to_string(index=False))

    # concat - stack DataFrames vertically
    q1 = df[df["date"].dt.quarter == 1].copy() if "date" in df.columns and hasattr(df["date"].dt, "quarter") else df.iloc[:50].copy()
    q2 = df[df["date"].dt.quarter == 2].copy() if "date" in df.columns and hasattr(df["date"].dt, "quarter") else df.iloc[:50:100].copy()
    q1["label"] = "Q1"
    q2["label"] = "Q2"
    combined = pd.concaat([q1, q2], ignore_index=True)
    print(f"\n Concat Q1+Q2 rows: {len(combined)}")

    return merged

#--------------------------------------------------------------
# SECTION 9: Apply & Lambda on DataFrames
# Concepts: .apply, .map(), .vectorized custom logic
#--------------------------------------------------------------

def apply_demo(df):
    print("\n ================ Apply & Map ================ ")

    # .apply() -> run a function on each row or column
    def revenue_tier(rev):
        if rev   >= 5000: return "Platinum"
        elif rev >= 2000: return "Gold"
        elif rev >= 500: return "Silver"
        else: return "Bronze"

    df["tier"] = df["revenue"].apply(revenue_tier)
    print(f"\n Revenue Tier Distribution: ")
    print(df["tier"].value_counts().to_string())

    # Lambda with apply - one-liner transformations
    df["discount_%"] = df["discount"].apply(lambda x: f"{x*100:.0f}%")

    # .map() -> map values through a dictionary
    region_zone = {"North", "Zone-1", "South", "Zone-2", "East", "Zone-3", "West", "Zone-4"}
    df["zone"] = df["region"].map(region_zone)

    # np.where -> vectorized if/else (much faster than, .apply for simple conditions)
    df["is_high_value"] = np.where(df["revenue"] > df["revenue"].median(), 1, 0)

    print(f"\n High-value orders: {df['is_high_value'].sum()} of {len(df)}")
    return df

#--------------------------------------------------------------
# SECTION 10: Correlation & Basic Stats
# Concepts: corr(), value_counts(), describe()
#      Direct precursor to ML feature selection
#--------------------------------------------------------------

def correlation_analysis(df):
    print("\n ============== Correlation Analysis ============== ")

    numeric_df = df[["units", "unit_price", "discount", "revenue"]].copy()

    corr_matrix = numeric_df.corr().round(3)
    print(f"\n Correlation Matrix! ")
    print(corr_matrix.to_string())

    print(f"\n Revenue Correlations (sorted): ")
    rev_corr = corr_matrix["revenue"].drop("revenue").sort_values(ascending=False)
    for col, val in rev_corr.items():
        bar = "$" * int(abs(val) * 20)
        direction = "+" if valu > 0 else "-"
        print(f"  {col:<12} {direction}{abs(val):.3f} {bar}")
    
    # Value counts - category distribution 
    print(f"\n Orders per Category: ")
    print(df["category"].value_counts().to_string())

    print(f"\n Return rate by Category: ")
    return_rate = df.groupby("category")["returned"].mean().sort_values(ascending=False)
    for cat, rate in return_rate.items():
        print(f" {cat:<15} {rate*100:.1f}%")

#--------------------------------------------------------------
# SECTION 11: Export
# Concepts: to_csv, to_json, to_excel
#--------------------------------------------------------------

def export_data(df):
    print("\n ============== Exporting Data ============== ")

    df.to_csv("sales_data.csv", index=False)
    print(" Exported to sales_data.csv")

    df.head(50).to_json("sales_sample.json", orient="records", indent=2)
    print(" Exported to sales_samples.json")

    summary = df.groupby("category").agg(
        orders       = ("order_id", "count"),
        revenue      = ("revenue", "sum"),
        avg_discount = ("discount", "mean")
    ).round(2)
    summary.to_csv("category_summary.csv")
    print(" Exported to category_summary.csv")

#----------------------------------------------------------
# SECTION 12: Main Menu
#-----------------------------------------------------------

def main():
    print("\n" + "="* 50)
    print(" DATA ANALYSIS WITH PANDAS & NUMPY ")
    print("="*50)

    print("\n Generating 200- row sales dataset.....")
    df = generate_sales_dataset(200)
    print(f" Dataset ready: {df.shape[0]} rows x {df.shape[1]} columns")

    while True:
        print("""
---------------- MAIN MENU ------------------------------------
              [1]  Numpy Fundamentals
              [2]  DataFrame Basics (shape, dtypes, describe)
              [3]  Data Cleaning (nulls, duplicates, datetime)
              [4]  Filtering & Selection (loc, iloc, query)
              [5]  GroupBy & Aggregation
              [6]  Pivot Tables
              [7]  Merging Dataframes (join, concat)
              [8]  Apply, Map & Lambda
              [9]  Correlation Analysis
              [10] Export Data (CSV, JSON)
              [11] Run Full Analysis Pipeline
              [0]  Exit
-------------------------------------------------------------""")
        
        choice = input(" Enter Choice: ").strip()

        try:
            if choice == "1":
                numpy_basics()
            
            elif choice == "2":
                dataframe_basics(df)

            elif choice == "3":
                df = clean_data(df)
            
            elif choice == "4":
                filtering_demo(df)
            
            elif choice == "5":
                df = groupby_analysis(df)
            
            elif choice == "6":
                # Ensure date column is datetime for pivot
                if df["date"].dtype == object:
                    df["date"] = pd.to_datetime(df["date"])
                pivot_analysis(df)
            
            elif choice == "7":
                merge_demo(df)
            
            elif choice == "8":
                df = apply_demo(df)
            
            elif choice == "9":
                correlation_analysis(df)
            
            elif choice == "10":
                export_data(df)              
            
            elif choice == "11":
                print("\n Running Full Analysis Pipeline...\n")
                numpy_basics()
                dataframe_basics(df)
                df = clean_data(df)
                filtering_demo(df)
                pivot_analysis(df)
                merge_demo(df)
                df = apply_demo(df)
                correlation_analysis(df)
                export_data(df)
                print("\n Full pipeline complete! ")
            
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
