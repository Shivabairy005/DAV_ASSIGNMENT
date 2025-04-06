import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("food_ingredients_and_allergens.csv")
print("Original Data (first 5 rows):\n", df.head())

# Convert Main Ingredient column to NumPy array (Fixed type array)
main_ing_array = df['Main Ingredient'].fillna("Unknown").astype(str).values
print("Main Ingredient Array:\n", main_ing_array[:5])

# Fixed type array: lengths of each ingredient
main_ing_lengths = np.array([len(i) for i in main_ing_array], dtype=np.int32)
print("Main Ingredient Lengths:\n", main_ing_lengths[:5])

# Create Arrays
zeros = np.zeros((2, 2))
ones = np.ones((2, 2))
identity = np.eye(2)
print("Zeros:\n", zeros)
print("Ones:\n", ones)
print("Identity:\n", identity)

# Indexing
print("First Element Length:", main_ing_lengths[0])

# Slicing
print("First Five Lengths:", main_ing_lengths[:5])

# Reshape
reshaped = main_ing_lengths[:12].reshape((3, 4))
print("Reshaped (3x4):\n", reshaped)

# Concatenate and Split
concat = np.concatenate([main_ing_lengths[:3], main_ing_lengths[3:6]])
print("Concatenated:\n", concat)

split = np.split(main_ing_lengths[:6], 2)
print("Split:\n", split)

# Universal Functions
sqrt_lengths = np.sqrt(main_ing_lengths[:5])
print("Square Roots:\n", sqrt_lengths)

exp_lengths = np.exp(main_ing_lengths[:3])  # using small subset
print("Exponentials:\n", exp_lengths)

# Aggregation
sum_lengths = np.sum(main_ing_lengths)
mean_lengths = np.mean(main_ing_lengths)
print("Sum of Lengths:", sum_lengths)
print("Mean of Lengths:", mean_lengths)

# Broadcasting
broadcasted = main_ing_lengths[:5] + 5
print("Broadcasted (added 5):\n", broadcasted)

# Comparisons / Boolean Mask
long_ingredients_mask = main_ing_lengths > 10
long_ingredients = main_ing_array[long_ingredients_mask]
print("Long Ingredients (len > 10):\n", long_ingredients[:5])

# Fancy Indexing
indices = [0, 2, 4]
fancy = main_ing_array[indices]
print("Fancy Indexing Result:\n", fancy)

# Fast Sorting
sorted_lengths = np.sort(main_ing_lengths[:10])
argsorted_lengths = np.argsort(main_ing_lengths[:10])
print("Sorted Lengths:\n", sorted_lengths)
print("Argsorted Indexes:\n", argsorted_lengths)

# Partial Sorting
partitioned = np.partition(main_ing_lengths[:10], 3)
print("Partially Sorted (Partitioned):\n", partitioned)

# Structured Array
structured = np.array(
    list(zip(df['Food Product'].fillna("None").values, df['Main Ingredient'].fillna("None").values)),
    dtype=[('product', 'U50'), ('ingredient', 'U50')]
)
print("Structured Array Sample:\n", structured[:3])

# Record Array View
record = structured.view(np.recarray)
print("Record Array Sample:\n", record[0].product, record[0].ingredient)

# Load the dataset

# --- Series Object ---
main_ing_series = pd.Series(df['Main Ingredient'].fillna("Unknown").values)
print("\nMain Ingredient Series:\n", main_ing_series.head())

# --- DataFrame Object ---
print("\nDataFrame Info:")
print(df.info())

# --- Data Indexing and Selecting for Series ---
print("\nAccessing Series Value by Position [0]:", main_ing_series[0])
print("Series Slice [0:3]:\n", main_ing_series[0:3])

# --- Data Indexing and Selecting for DataFrame ---
print("\nSelecting Single Column (Food Product):\n", df['Food Product'].head())
print("\nSelecting Multiple Columns:\n", df[['Food Product', 'Main Ingredient']].head())

# --- Universal Functions for Index Preservation ---
upper_ing = main_ing_series.str.upper()
print("\nUppercase Ingredients:\n", upper_ing.head())

# --- Index Alignment and Operations ---
new_series = pd.Series([1, 2, 3], index=[0, 1, 2])
added_series = new_series + pd.Series([10, 20], index=[1, 2])
print("\nIndex Aligned Addition:\n", added_series)

# --- Handling Missing Data ---
null_counts = df.isnull().sum()
print("\nMissing Values in Each Column:\n", null_counts)

df_filled = df.fillna("Missing")
print("\nAfter Filling Missing Values (sample):\n", df_filled.head())

# --- Operating on Null Values ---
null_mask = df['Main Ingredient'].isnull()
print("\nRows with Null 'Main Ingredient':\n", df[null_mask].head())

# --- Hierarchical Indexing ---
# Create a new small DataFrame for this example
sample_df = df[['Food Product', 'Main Ingredient']].dropna().copy()
sample_df = sample_df.set_index(['Main Ingredient', 'Food Product'])
print("\nHierarchically Indexed DataFrame:\n", sample_df.head())

# Split dataset into two parts and then concatenate
# --- CONCAT ---
df1 = df.iloc[:5]
df2 = df.iloc[5:10]
concatenated = pd.concat([df1, df2])
print("\n🔹 After Concatenation (df1 + df2):\n", concatenated)

# ---------- APPEND ----------
appended = pd.concat([df1, df2], ignore_index=True)
print("\n🔹 After Append (df1 + df2 using concat):\n", appended)

# ---------- MERGE ----------
ingredients_info = pd.DataFrame({
    "Main Ingredient": ["Salt", "Sugar", "Wheat"],
    "Category": ["Mineral", "Sweetener", "Grain"]
})
merged = pd.merge(df, ingredients_info, how="left", on="Main Ingredient")
print("\n🔹 After Merge with ingredient category info:\n", merged[['Main Ingredient', 'Category']].dropna().head())

# ---------- JOIN ----------
left = df.set_index("Food Product").iloc[:5]
right = df.set_index("Food Product")[["Main Ingredient"]].iloc[5:10]
joined = left.join(right, lsuffix='_left', rsuffix='_right', how='outer')
print("\n🔹 After Join:\n", joined)

# ---------- GROUPBY ----------
grouped = df.groupby("Main Ingredient").count()
print("\n🔹 Grouped by 'Main Ingredient' (count of rows per ingredient):\n", grouped[['Food Product']].head())

# ---------- AGGREGATION ----------
agg_data = df.groupby("Main Ingredient").agg({
    "Food Product": "count",
    "Allergens": lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"
})
print("\n🔹 Aggregated Data (count + most common Allergen):\n", agg_data.head())

# ---------- PIVOT TABLE ----------
pivot = df.pivot_table(index="Main Ingredient", columns="Allergens", values="Food Product", aggfunc="count")
pivot_cleaned = pivot.fillna(0).astype(int)
print("\n🔹 Pivot Table (count of Food Products per Ingredient per Allergen):\n", pivot_cleaned.head())

plt.figure(figsize=(12, 6))
subset = pivot_cleaned.iloc[:10, :5]  # first 10 ingredients and 5 allergens
plt.imshow(subset, cmap='Blues', aspect='auto')
plt.colorbar(label='Count')
plt.xticks(ticks=range(len(subset.columns)), labels=subset.columns, rotation=45)
plt.yticks(ticks=range(len(subset.index)), labels=subset.index)
plt.title("Pivot Table Heatmap: Ingredients vs Allergens")
plt.tight_layout()
plt.show()

top_ingredients = df['Main Ingredient'].value_counts().head(5)
plt.figure(figsize=(7,7))
plt.pie(top_ingredients, labels=top_ingredients.index, autopct='%1.1f%%', startangle=140)
plt.title("Top 5 Main Ingredients in Food Products")
plt.axis('equal')
plt.show()

ingredient_counts = df['Main Ingredient'].value_counts().sort_values(ascending=False).head(15)
plt.figure(figsize=(10,5))
plt.plot(ingredient_counts.index, ingredient_counts.values, marker='o', linestyle='-')
plt.title("Top 15 Ingredient Frequencies")
plt.xlabel("Main Ingredient")
plt.ylabel("Number of Occurrences")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()