"""
Project 7: Data Visualization Dashboard
Concepts: matplotlib, seaborn, subplots, figure/axes,
          bar/line/scatter/pie/heatmap/boxplot/histogram,
          styling, annotations, saving figures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings("ignore")

#-------------------------------------------------------------------
# SECTION 1: Style Configuration
# Concepts: Global plot styling -- applied once, affects all charts
#-------------------------------------------------------------------

def set_style():
    """ Configure global matplotlib & seaborn style. """
    plt.rcParams.update({
        "figure.facecolor"      : "#0f0f0f",        #dark background
        "axes.facecolor"        : "#1a1a2e",
        "axes.edgecolor"        : "#444",
        "axes.labelcolor"       : "#cccccc",
        "axes.titlecolor"       : "#ffffff",
        "axes.titlesize"        : 13,
        "axes.labelsize"        : 11,
        "xtick.color"           : "#aaaaaa",
        "ytick.color"           : "#aaaaaa",
        "text.color"            : "#cccccc",
        "grid.color"            : "#2a2a3e",
        "grid.linestyle"        : "--",
        "grid.alpha"            : 0.5,
        "legend.facecolor"      : "#1a1a2e",
        "legend.edgecolor"      : "#444",
        "legend.fontsize"       : 9,
        "font.family"           : "DejaVu Sans",
    })
    # Seaborn palette
    sns.set_palette("bright")

#--------------------------------------------------------------
# SECTION 2:Dataset Generator (reused from Project 6)
#--------------------------------------------------------------

def generate_dataset(n=300):
    """
    Generate a synthetic dataset.
    Returns a pandas DataFrame -- the core data structure.
    """
    random.seed(42)
    np.random.seed(42)

    regions     = ["North", "South", "East", "West"]
    categories  = ["Electronics", "Clothing", "Food", "Books", "Sports"]
    products    = {
        "Electronics": ["Laptop", "Phone", "Table", "Headphones"],
        "Clothing"  : ["Shirts", "Jeans", "Jacket", "Shoes"],
        "Food"      : ["Croceries", "Snacks", "Beverages", "Frozen"],
        "Books"     : ["Fiction", "Non-Fiction", "Textbook", "Comics", "Novels"],
        "Sports"    : ["Football", "Cricket Kit", "Gym Equipment", "Yoga Mat"]
    }
    salespeople = ["Ali", "Sara", "Raza", "Ayesha", "Bilal", "Hina", "Omar", "Zara"]

    start = datetime(2027, 1, 1)
    rows = []

    for i in range(n):
        cat         = random.choice(categories)
        region      = random.choice(regions)
        sp          = random.choice(salespeople)
        units       = random.randint(1, 50)
        price       = round(random.uniform(10, 500), 2)
        disc        = round(random.uniform(0, 0.3), 2)  # 0-30% discount
        revenue     = round(units * price * (1 - disc), 2)
        date        = start + timedelta(days=random.randint(0, 364))

        rows.append({
            "date"          : date,
            "month"         : date.strftime("%B"),
            "month_num"     : date.month,
            "quater"        : f"Q{(date.month-1)//3 + 1}",
            "region"        : region,
            "salesperson"   : sp,
            "category"      : cat,
            "product"       : random.choice(products[cat]),
            "units"         : units,
            "unit_price"    : price,
            "discount"      : disc,
            "revenue"       : revenue,
            "returned"      : random.random() < 0.05,
        })
    
    df = pd.DataFrame(rows)             # list of dicts - DataFrame
    df = df.sort_values("date").reset_index(drop=True)
    return df

#--------------------------------------------------------------
# SECTION 3: Helper -- Save & Show 
#--------------------------------------------------------------

def save_show(fig, filename):
    """ Save figure to file and dispaly it. """
    fig.savefig(filename, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f" Saved: {filename}")
    plt.show()
    plt.close(fig)

#--------------------------------------------------------------
# SECTION 4: Chart 1 -- Bar Charts
# Concepts: fig, ax, bar(), barh(), annotating bars
#--------------------------------------------------------------

def plot_bar_charts(df):
    """ Vertical and horizontal bar charts side by side. """
    # Cocept: fig, ax = plt.subplots() -- always use this pattern
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Bar Charts -- Sales by Region and Category ", fontsize=15, color="white", y=1.02)

    colors = ["#e94560", "#0f3460", "#533483", "#05c46b"]

    # ---- Vertical bar chart ----
    region_rev = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    bars = axes[0].bar(region_rev.index, region_rev.values, color=colors, edgecolor="#333", linewidth=0.8)
    axes[0].set_title("Revenue by Region")
    axes[0].set_xlabel("Region")
    axes[0].set_ylabel("Revenue ($)")
    axes[0].grid(axis="y", alpha=0.3)

    # Annotate each bar with its value
    for bar in bars:
        height = bar.get_height()
        axes[0].annotate(
            f"${height/1000:.1f}K",
            xy          = (bar.get_x()+ bar.get_width() / 2, height),
            xytext      = (0, 5),
            textcoords  = "offset points",
            ha          = "center",
            va          = "bottom",
            fontsize    = 9, 
            color       = "white"
        )
    
    # --- Horizontal bar chart ---
    cat_rev     = df.groupby("category")["revenue"].sum().sort_values()
    cat_colors  = ["#e94560", "#0f3460", "#533483", "#05c46b", "#ffd460"]
    bars_h      = axes[1].barh(cat_rev.index, cat_rev.values, color=cat_colors, edgecolor="#333")
    axes[1].set_title("Revenue by Category")
    axes[1].set_xlabel("Revenue ($)")
    axes[1].grid(axis="x", alpha=0.3)

    for bar in bars_h:
        width = bar.get_width()
        axes[1].annotate(
            f"${width/1000:.1f}K",
            xy          =(width, bar.get_y() + bar.get_height() / 2),
            xytext      = (3,0),
            textcoords  = "offset points",
            ha          = "left",
            va          = "center",
            fontsize    = 9,
            color       = "white"
        )
    
    plt.tight_layout()
    save_show(fig, "chart1_bar.png")

#----------------------------------------------------------------------
# SECTION 5: Chart 2 -- Live Charts
# Concepts: plot(), datetime x-axis, multiple lines
#           rolling average (moving average  -- used in time series AI)
#----------------------------------------------------------------------

def plot_line_charts(df):
    """ Monthly revenue trends with rolling average"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("Line Charts - Remove Trends Over Time", fontsize=15, color="white")

    # --- Monthly total revenue ---
    monthly = df.groupby("month_num")["revenue"].sum().reset_index()
    monthly.columns = ["month", "revenue"]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly["label"] = monthly["month"].apply(lambda m: month_labels[n-1])

    axes[0].plot(monthly["month"], monthly["revenue"],
                 color              = "#e94560",
                 linewidth          = 2.5,
                 marker             = "o",
                 markersize         = 7,
                 markerfacecolor    = "white",
                 label              = "Monthly Revenue"
                 )

    # Rolling average -- smooths out noise (key concept in time series)
    rolling = monthly["revenue"].rolling(window=3, center=True).mean()
    axes[0].plot(monthly["month"], rolling,
                 color      = "#ffd460",
                 linewidth  = 2,
                 linestyle  = "--",
                 alpha      = 0.8,
                 label      = "3-Month Rolling Average"                                  
                 )
    # Fill area under line
    axes[0].fill_between(monthly["month"], monthly["revenue"], alpha=0.15, color="#e94560")
    
    axes[0].set_title("Monthly Revenue + 3-Month Rolling Average")
    axes[0].set_xticks(monthly["month"])
    axes[0].set_xtickslabels(monthly["label"])
    axes[0].set_ylabel("Revenue ($)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Annotate peak month
    peak_idx = monthly["revenue"].idxmax()
    peak_val = monthly["revenue"].max()
    peak_mon = monthly.loc[peak_idx, "label"]
    axes[0].annotate(
        f"Peak: {peak_mon}\n${peak_val/1000:.1f}K",
        xy          = (monthly.loc[peak_idx, "month"], peak_val),
        xytext      = (20, -30), textcoords="offset points",
        arrowprops  = dict(arrowstyle = "->", color = "white"),
        color       = "white",
        fontsize    = 9,
        bbox        = dict(boxstyle="round, pad=0.3", facecolor = "#333", edgecolor="#666")
    )

    # -- Multi-line: revenue per category per month --
    cat_monthly = df.groupby(["month_num", "category"])["revenue"].sum().unstack(fill_value=0)
    line_colors = ["#e94560", "#0f3460", "#533483", "#05c46b", "#ffd460"]

    for (cat, color) in zip(cat_monthly.columns, line_colors):
        axes[1].plot(cat_monthly.index, cat_monthly[cat],
                     label      = cat,
                     color      = color,
                     linewidth  = 2,
                     marker     = ".",
                     markersize = 5
                     )
    
    axes[0].set_title("Monthly Revenue by Category")
    axes[0].set_xticks(range[1, 13])
    axes[0].set_xtickslabels(month_labels)
    axes[0].set_ylabel("Revenue ($)")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    plt.tight_layout()
    save_show(fig, "chart2_line.png")

#--------------------------------------------------------------
# SECTION 6: Chart 3 - Scatter Plots
# Concepts: scatter(), color-coding by category,
#           size encoding, trend line (polyfit)
#--------------------------------------------------------------

def plot_scatter(df):
    """ Scatter plot: units vs revenue, color by category. """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(" Scatter Plots -- Relationships in Data", fontsize=15, color="white")

    categories = df["category"].unique()
    colors = ["#e94560", "#0f3460", "#533483", "#05c46b", "#ffd460"]
    cat_color = dict(zip(categories, colors))

    # --- Scatter: units vs revenue ---
    for cat in categories:
        subset = df[df["category"] == cat]
        axes[0].scatter(subset["units"], subset["revenue"],
                        c           = cat_color[cat],
                        label       = cat,
                        alpha       = 0.6,
                        s           = 40,
                        edgecolors  = "none"
                        )
    
    # Trend line -- np.apply.polyfit fits a polynomial (dgree 1 = linear regression!)
    z = np.polyfit(df["units"], df["revenue"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df["units"].min(), df["units"].max(), 100)
    axes[0].plot(x_line, p(x_line),
                 color      = "white",
                 linewidth  = 1.5,
                 linestyle  = "--",
                 alpha      = 0.7,
                 label      = "Trend Lie"
                 )
    
    axes[0].set_title("Units Sold vs Revenue")
    axes[0].set_xlabel("Units Sold")
    axes[0].set_ylabel("Revenue ($)")
    axes[0].legend(fontsize=0)
    axes[0].grid(alpha=0.2)

    # -- Scatter: discount vs revenue, size = units --
    scatter = axes[1].scatter(
        df["discount"], df["revenue"],
        c           = df["units"],          # color encodes units
        cmap        = "plasma",             # colormap
        s           = df["units"] * 2,      # size encodes units
        alpha       = 0.5,
        edgecolors  = "none"
    )
    plt.colorbar(scatter, ax=axes[1], label="Units Sold")
    axes[1].set_title(" Dicount vs Revenue \n (color & size = units sold)")
    axes[1].set_xlabel("Discount Rate")
    axes[1].set_ylabel("Revenue ($)")
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    save_show(fig, "chart3_scatter.png")

#--------------------------------------------------------------
# SECTION 7: Chart 4 -- Pie & Donut Charts
# Concepts: pie(), wedgeprops for donut, explode
#--------------------------------------------------------------

def plot_pie_charts(df):
    """ Pie and donut charts for proportional data. """
    fig, axes = plt.subplots(1,2, figsize=(14, 7))
    fig.suptitle(" Pie & Donut Charts -- Share of revenue", fontsize=15, color="white")

    colors = ["#e94560", "#0f3460", "#533483", "#05c46b", "#ffd460"]

    # -- Pie chart: revenue by category --
    cat_rev = df.groupby("category")["revenue"].sum()
    explode = [0.05] * len(cat_rev)         #slightly separate all slices

    axes[0].pie(
        cat_rev.values,
        labels = cat_rev.index,
        colors = colors,
        explode = explode,
        autopct = "%1.1f%%",
        startangle = 140,
        textprops = {"color": "white", "fontsize": 9},
        wedgeprops = {"edgecolor": "#0f0f0f", "linewidth": 1.5}
    )
    axes[0].set_title("Revenue Share by Category")

    # --- Donut chart: revenue by region ---
    region_rev = df.groupby("region")["revenue"].sum()
    region_colors = ["#e94560", "#533483", "#05c46b", "#ffd460"]

    wedges, texts, autotexts = axes[1].pie(
        region_rev.values,
        labels = region_rev.index,
        colors = region_colors,
        autopct = "%1.1f%%",
        startangle = 90,
        textprops = {"color": "white", "fontsize": 9},
        wedgeprops = {"edgecolor": "#0f0f0f", "linewidth": 2, "width": 0.6} # width < 1 creates donut hole
    )

    # Add center text
    total = region_rev.sum()
    axes[1].text(0,0, f"Total\n${total/1000:.0f}K",
                 ha         = "center",
                 va         = "center",
                 fontsize   = 11,
                 color      = "white",
                 fontweight = "bold"
                 )
    axes[1].set_title("Revenue Share by Region (Donut)")

    plt.tight_layout()
    save_show(fig, "chart4_pie.png")

#--------------------------------------------------------------
# SECTION 8: Chart 5 -- Histogram & KDE
# Concepts: hist(), seaborn kdeplot, distribution analysis
#--------------------------------------------------------------

def plot_distributions(df):
    """ Histograms and KDE plots for distribution analysis. """
    fig, axes = plt.subplots(2,2, figsize=(14, 10))
    fig.suptitle(" Distributions -- Histogram & KDE", fontsize=15, color="white")

    # ---- Revenue histogram ----
    axes[0,0].hist(df["revenue"], bins=30, color="#e94560", edgecolor="#0f0f0f",
                   alpha=0.8, linewidth=0.5)
    axes[0,0].axvline(df["revenue"].mean(), color="yellow", linestyle="--",
                      linewidth=1.5, label=f"Mean: ${df['revenue'].mean():.0f}")
    axes[0,0].set_title("Revenue Distribution")
    axes[0,0].set_xlabel("Revenue ($)")
    axes[0,0].legend()
    axes[0,0].grid(alpha=0.3)

    # ---- KDE plot (smooth distribution curve) ----
    for cat, color in zip(df["category"].unique(), ["#e94560", "#533483", "#05c46b", "#ffd460", "#0f3460"]):
        subset = df[df["category"] == cat]["revenue"]
        sns.kdelot(subset, ax=axes[0,1], color=color, label=cat, linewidth=2, fill=True, alpha=0.1)
    axes[0,1].set_title("Revenue KDE by Category")
    axes[0,1].set_xlabel("Revenue ($)")
    axes[0,1].legend(fontsize=0)
    axes[0,1].grid(alpha=0.3)

    # ---- Units histogram ----
    axes[1,0].hist(df["units"], bins=25, color="#533483", edgecolor="#0f0f0f",
                   alpha=0.8, linewidth=0.5)
    axes[1,0].set_title("Units Sold Distribution")
    axes[1,0].set_xlabel("Units")
    axes[1,0].grid(alpha=0.3)

    # ---- Discount distribution ----
    axes[1,1].hist(df["discount"] * 100, bins=20, color="#05c46b", edgecolor="#0f0f0f",
                   alpha=0.8, linewidth=0.5)
    axes[1,1].set_title("Discount % Distribution")
    axes[1,1].set_xlabel("Discount (%)")
    axes[1,1].grid(alpha=0.3)

    plt.tight_layout()
    save_show(fig, "chart5_distributions.png")

#--------------------------------------------------------------
# SECTION 9: Chart 6 -- Heatmaps
# Concepts: seaborn heatmap, correlation matrix, 
#           pivot heatmap -- most used chart in ML EDA
#--------------------------------------------------------------

def plot_heatmaps(df):
    """ Correlation heatmap and pivot heatmap. """
    fig, axes = plt.subplots(1,2, figsize=(16, 7))
    fig.suptitle(" Heatmaps -- Correlations & Pivot", fontsize=15, color="white")

    # ---- Correlation Heatmap ----
    numeric_cols = ["units", "unit_price", "discount", "revenue"]
    corr         = df[numeric_cols].corr()

    sns.heatmap(
        corr,
        ax          = axes[0],
        annot       = True,         # show numbers in cells
        fmt         = ".2f",        # 2 decimal places
        cmap        = "coolwarm",   # diverging colormap: blue=negative, red=positive
        center      = 0,            # center colormap at 0
        linewidths  = 0.5,
        linecolor   = "#0f0f0f",
        cbar_kws    = {"shrink": 0.8}
    )
    axes[0].set_title("Correlation Matrix")

    # ---- Pivot Heatmap: revenue by region x Category ----
    pivot = df.pivot_table(values="revenue", index="region",
                           columns="category", aggfunc="sum")
    pivot_normalized = pivot.div(pivot.sum(axis=1), axis=0)  # normalize by row

    sns.heatmap(
        pivot_normalized,
        ax          = axes[1],
        annot       = True,         # show numbers in cells
        fmt         = ".1f",        # shows as percentage
        cmap        = "Y10rRd",   
        linewidths  = 0.5,
        linecolor   = "#0f0f0f",
        cbar_kws    = {"shrink": 0.8}
    )
    
    axes[1].set_title("Revenue Share: Region x Category\n(row-normalized)")
    
    plt.tight_layout()
    save_show(fig, "chart6_heatmaps.png")

#--------------------------------------------------------------
# SECTION 10: Chart 7 -- Box plots & Violin Plots
# Concepts: seaborn boxpolt, violinplot - outlier detection,
#           distribution comparision across groups
#--------------------------------------------------------------

def plot_box_violin(df):
    """ Box and violin plots for distribution comparions."""
    fig, axes = plt.subplots(1,2, figsize=(14, 7))
    fig.suptitle(" Box & Violin Plots - Distribution Comparisons", fontsize=15, color="white")

    # ---- Box plot: revenue per category ----
    # Shows: median, IQR (box), whiskers, outliers (dots)
    sns.boxplot(
        data        = df,
        x           = "category",
        y           = "revenue",
        palette     = "bright",
        ax          = axes[0],
        flierprops  = {"markerfacecolor": "red", "markersize":4}
    )
    axes[0].set_title("Revenue Distribution by Category\n(Box Plot)")
    axes[0].set_xlabel("Category")
    axes[0].set_ylabel("Revenue ($)")
    axes[0].tick_params(axis="X", rotation=15)
    axes[0].grid(axis="y", alpha=0.3)

    # ---- Violin plot: revenue per region ---- 
    # Combines box plot + KDE -- shows full distribtion shape
    sns.violinplot(
        data        = df,
        x           = "region",
        y           = "revenue",
        palette     = ["#e94560", "#ffd460" "#533483", "#05c46b"],
        ax          = axes[1],
        inner       = "quartile"        #shows quartile lines inside violin
    )
    axes[1].set_title("Revenue Distribution by Region\n(Violin Plot)")
    axes[1].set_xlabel("Region")
    axes[1].set_ylabel("Revenue ($)")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_show(fig, "chart7_box_violin.png")


#--------------------------------------------------------------
# SECTION 11: Chart 8 -- Full Dashboard
# Concepts: GridSpec for complex multi-panel layouts
#           This is how Jupyter notebooks looks in ML projects
#--------------------------------------------------------------

def plot_dashboard(df):
    """ Full executive dashboard - 6 panels in one figure. """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(" Salex Executive Dashboard - 2026",
                 fontsize=18, color="white", fontweight="bold", y=0.98)

    # GridSpec: 3 rows x 4 columns - gives fine control over layout
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.4)

    ax1 = fig.add_subplot(gs[0, :2]) # row 0, cols 0-1 (wide)
    ax2 = fig.add_subplot(gs[0, 2:]) # row 0, cols 2-3 (wide)
    ax3 = fig.add_subplot(gs[1, :2]) # row 1, cols 0-1
    ax4 = fig.add_subplot(gs[1, 2:]) # row 1, cols 2-3
    ax5 = fig.add_subplot(gs[2, :2]) # row 2, cols 0-1
    ax6 = fig.add_subplot(gs[2, 2:]) # row 2, cols 2-3

    colors = ["#e94560", "#0f3460", "#533483", "#05c46b", "#ffd460"]

    # -- Panel 1: Monthly Revenue Line --
    monthly = df.groupby("month_num")["revenue"].sum()
    months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    ax1.plot(range(1, 13), [monthly.get(m, 0) for m in range(1,13)],
                   color="#e94560", linewidth=2.5, marker="0", markersize=5,
                   markerfacecolor="white")
    ax1.fill_between(range(1, 13), [monthly.get(m, 0) for m in range(1, 13)],
             alpha=0.15, color="#e94560")
    ax1.set_xticks(range(1,13)); ax1.set_xticklabels(months)
    ax1.set_title("Monthly Revenue"); ax1.grid(alpha=0.3)
    
    # -- Panel 2: Revenue by Category Bar --
    cat_rev = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
    ax2.bar(range(len(cat_rev)), cat_rev.values, color=colors, edgecolor="#333")
    ax2.set_xticks(range(len(cat_rev)))
    ax2.set_xticklabels(cat_rev.index, rotation=15, fontsize=8)
    ax2.set_title("Revenue by Category"); ax2.grid(axis="y", alpha=0.3)

    # -- Panel 3: Region Donut --
    region_rev = df.groupby("region")["revenue"].sum()
    ax3.pie(region_rev.values, labels=region_rev.index,
            colors=["#e94560", "#533483", "#05c46b", "#ffd460"],
            autopct="%1.0f%%", startangle=90,
            textprops={"color":"white","fontsize":8},
            wedgeprops={"width":0.55, "edgecolor":"#0f0f0f"})
    ax3.set_title("Region Share (Donut)")

    # -- Panel 4: Correlation Heatmap --
    corr = df[["units", "units_price", "discount", "revenue"]].corr()
    sns.heatmap(corr, ax=ax4, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, linecolor="#0f0f0f",
                cbar=False, annot_kws={"size":0})
    ax4.set_title("Correlation Matrix")

    # -- Panel 5: Top Salespeople Bar --
    sp_rev      = df.groupby("salesperson")["revenue"].sum().sort_values(ascending=True)
    sp_colors   = [colors[i % len(colors)] for i in range(len(sp_rev))]
    ax5.barh(sp_rev, sp_rev.values, colors=sp_colors, edgecolor="#333")
    ax5.set_title("Revenue by Salesperson"); ax5.grid(axis="x", alpha=0.3)

    # -- Panel 6: Revenue Distribution Histogram --
    ax6.hist(df["revenue"], bins=25, color="#533483", edgecolor="#0f0f0f",
             alpha=0.85, linewidth=0.5)
    ax6.axvline(df["revenue"].mean(), color="yellow", linestyle="--",
                linewidth=1.5, label=f"Mean")
    ax6.set_title("Revenue Distribution")
    ax6.legend(); ax6.grid(alpha=0.3)

    # KPI strip at very top
    kpi_y = 0.93
    total_rev   = df["revenue"].sum()
    avg_rev     = df["revenue"].mean()
    total_ord   = len(df)
    top_cat     = df.groupby("category")["revenue"].sum().idmax()

    kpis = [
        f" Total Revenue: ${total_rev/1000:.1f}K",
        f" Total Orders: {total_ord}",
        f" Avg Order: ${avg_rev:.0f}",
        f" Top Category: {top_cat}",
    ]
    for i, kpi in enumerate(kpis):
        fig.text(0.04 + i * 0.24, kpi_y, kpi,
                 fontsize=10, color="#ffd460", fontweight="bold",
                 transform=fig.transFigure)
    
    save_show(fig, "chart8_dashboard.png")

#----------------------------------------------------------
# SECTION 12: Main Menu
#-----------------------------------------------------------

def main():
    print("\n" + "="* 52)
    print(" DATA VISUALIZATION DASHBOARD ")
    print("="*52)

    set_style()

    print("\n Generating dataset.....")
    df = generate_dataset(300)
    print(f" Dataset ready: {df.shape[0]} rows x {df.shape[1]} columns")

    while True:
        print("""
---------------- MAIN MENU ------------------------------------
              [1]  Bar Charts           (region & category revenue)
              [2]  Line Charts          (monthly trends + rolling avg)
              [3]  Scatter Plots        (units vs revenue + trend line)
              [4]  Pie & Donut Charts   (revenue shares)
              [5]  Histogram & KDE      (distributions)
              [6]  Heatmaps             (correlation + pivot)
              [7]  Box & Violin Plots   (outliers + spread)
              [8]  Full Dashboard       (6-panel executive view)
              [9]  Generate All Charts
              [0]  Exit
-------------------------------------------------------------""")
        
        choice = input(" Enter Choice: ").strip()

        try:
            if choice == "1":
                plot_bar_charts(df)
            elif choice == "2":
                plot_line_charts(df)
            elif choice == "3":
                plot_scatter(df)
            elif choice == "4":
                plot_pie_charts(df)
            elif choice == "5":
                plot_distributions(df)
            elif choice == "6":
                plot_heatmaps(df)
            elif choice == "7":
                plot_box_violin(df)
            elif choice == "8":
                plot_dashboard(df)
            elif choice == "9":
                print("\n Generating All Charts! ...\n")
                for fn in [plot_bar_charts, plot_line_charts, plot_scatter,
                           plot_pie_charts, plot_distributions, plot_heatmaps,
                           plot_box_violin, plot_dashboard]:
                    fn(df)
                print("\n All Charts Saved as PNG files! ")
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