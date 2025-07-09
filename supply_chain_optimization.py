import pandas as pd
import pulp
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
data = pd.read_csv('train.csv')

# Convert weight from grams to kilograms
data['Weight_in_kg'] = data['Weight_in_gms'] / 1000

# Define cost per kg for each mode of shipment
cost_per_kg = {
    'Flight': 20,
    'Ship': 10,
    'Road': 5
}

# Group total weight by Warehouse_block and Mode_of_Shipment
grouped_data = data.groupby(['Warehouse_block', 'Mode_of_Shipment'])['Weight_in_kg'].sum().reset_index()
total_actual_weight = data['Weight_in_kg'].sum()

# Create a linear programming problem
problem = pulp.LpProblem("Balanced_Shipping_Optimization", pulp.LpMinimize)

# Decision variables - weight to be allocated to each (warehouse, mode) pair
decision_vars = pulp.LpVariable.dicts(
    "Weight", 
    [(row.Warehouse_block, row.Mode_of_Shipment) for _, row in grouped_data.iterrows()],
    lowBound=0
)

# Objective function: Minimize total shipping cost
problem += pulp.lpSum(
    decision_vars[(row.Warehouse_block, row.Mode_of_Shipment)] * cost_per_kg[row.Mode_of_Shipment]
    for _, row in grouped_data.iterrows()
)

# Constraints -----------------------------------------------------------------

# 1. Total weight equal to actual weight
problem += pulp.lpSum(decision_vars.values()) == total_actual_weight

# 2. Minimum 20% allocation to each mode (to prevent single mode domination)
for mode in cost_per_kg.keys():
    problem += pulp.lpSum(
        var for key, var in decision_vars.items()
        if key[1] == mode
    ) >= 0.2 * total_actual_weight, f"min_{mode}_allocation"

# 3. Maximum 50% allocation to any single mode (to force diversity)
for mode in cost_per_kg.keys():
    problem += pulp.lpSum(
        var for key, var in decision_vars.items()
        if key[1] == mode
    ) <= 0.5 * total_actual_weight, f"max_{mode}_allocation"

# 4. Respect original warehouse capacities
for warehouse in grouped_data['Warehouse_block'].unique():
    problem += pulp.lpSum(
        var for key, var in decision_vars.items()
        if key[0] == warehouse
    ) <= 1.1 * grouped_data[grouped_data['Warehouse_block'] == warehouse]['Weight_in_kg'].sum()

# Solve the optimization problem
problem.solve()

# Collect results -------------------------------------------------------------
results = []
for key, var in decision_vars.items():
    results.append({
        'Warehouse': key[0],
        'Mode': key[1],
        'Weight': var.varValue,
        'Cost': var.varValue * cost_per_kg[key[1]]
    })
results_df = pd.DataFrame(results)

# Visualization ---------------------------------------------------------------
plt.figure(figsize=(14, 8))

# 1. Warehouse-level allocation breakdown
plt.subplot(1, 2, 1)
modes = ['Flight', 'Ship', 'Road']
bottom = np.zeros(len(grouped_data['Warehouse_block'].unique()))

for mode in modes:
    weights = [results_df[(results_df['Warehouse'] == wh) & 
                         (results_df['Mode'] == mode)]['Weight'].sum() 
              for wh in sorted(grouped_data['Warehouse_block'].unique())]
    plt.bar(
        sorted(grouped_data['Warehouse_block'].unique()),
        weights,
        bottom=bottom,
        label=f'{mode} (₹{cost_per_kg[mode]}/kg)'
    )
    bottom += weights

plt.xlabel('Warehouse')
plt.ylabel('Weight (kg)')
plt.title('Weight Allocation by Warehouse')
plt.legend()
plt.grid(True, axis='y', alpha=0.3)

# 2. Mode-level cost analysis
plt.subplot(1, 2, 2)
mode_totals = results_df.groupby('Mode').agg({'Weight': 'sum', 'Cost': 'sum'})
colors = ['#FF7F0E', '#1F77B4', '#2CA02C']  # Flight, Ship, Road

plt.bar(
    mode_totals.index,
    mode_totals['Weight'],
    color=colors,
    alpha=0.7,
    label='Weight (kg)'
)
plt.ylabel('Weight (kg)')
plt.xlabel('Shipping Mode')
plt.title('Total Allocation by Shipping Mode')
plt.twinx()
plt.plot(
    mode_totals.index,
    mode_totals['Cost'],
    color='red',
    marker='o',
    label='Cost (₹)'
)
plt.ylabel('Cost (₹)')

# Add data labels
for i, (mode, row) in enumerate(mode_totals.iterrows()):
    plt.text(
        i,
        row['Weight'] * 1.05,
        f"{row['Weight']:.1f}kg\n₹{row['Cost']:.1f}",
        ha='center',
        va='bottom'
    )

plt.legend(loc='upper right')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()

# Print summary statistics
print("\n=== Optimization Results ===")
print(f"Total Weight: {total_actual_weight:.2f} kg")
print(f"Minimized Total Cost: ₹{pulp.value(problem.objective):.2f}\n")

print("=== Mode Distribution ===")
for mode, row in mode_totals.iterrows():
    print(f"{mode}: {row['Weight']:.1f} kg ({row['Weight']/total_actual_weight:.1%}) | Cost: ₹{row['Cost']:.2f}")

print("\n=== Warehouse Allocation ===")
warehouse_totals = results_df.groupby('Warehouse')['Weight'].sum()
for wh in sorted(warehouse_totals.index):
    wh_data = results_df[results_df['Warehouse'] == wh]
    print(f"\nWarehouse {wh}:")
    for _, row in wh_data.iterrows():
        print(f"  {row['Mode']}: {row['Weight']:.1f} kg")

plt.show()
