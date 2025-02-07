# Import necessary libraries

import pandas as pd  
import pyarrow as pa  
import pyarrow.parquet as pq  
import numpy as np  
from pulp import LpMaximize, LpProblem, LpVariable, lpSum  # For linear programming optimization

# --- Read Data ---

# Data informing DSEIs regions (Special Indigenous Sanitary Districts)
dsei = pd.read_csv('/Users/julianeoliveira/Documents/github/Surveillance_Network_Indigenous_Mobility/Input_data/dsei.csv')

# Data informing current Brazil's health sentinel network 
df_ms = pd.read_csv('/Users/julianeoliveira/Documents/github/Surveillance_Network_Indigenous_Mobility/Input_data/clean_df_ms.csv')

# Data of mobility coverage obtained by Oliveira et al.
df_mob = pd.read_csv('/Users/julianeoliveira/Documents/github/sentinel_spots_early_patho_detection/Data/first_level_of_mobility_coverage.csv')

# --- Process Data and Create Auxiliary Subdata ---

# Select relevant columns for DSEI dataset
dsei = dsei[['Municipio', 'idm_ioibge', 'Area_MUN', 'estado',
             'capital', 'populacao', 'NomeDSEI', 'StatusDSEI', 'NumDSEI']]

# Clean up DSEI data: fix names and remove unnecessary spaces
dsei['NomeDSEI'] = dsei['NomeDSEI'].replace(' Manaus', 'Manaus') 
dsei['NomeDSEI'] = dsei['NomeDSEI'].replace(' ', np.nan)  # Replace empty spaces with NaN

# Group the current sentinel network data by city, summing the active sentinel count
sent = df_ms.groupby(['UF', 'Nome_Município', 'Código Município Completo'])['ones'].sum().reset_index()

# Rename the column to clarify its purpose
sent = sent.rename(columns={'ones': 'number_active_sent'})

# Create a new column 'ones' to mark active sentinel locations
sent = sent.assign(ones=1)

# Join DSEI and sentinel network data based on city code
dsei2 = dsei.set_index('idm_ioibge').join(sent.set_index('Código Município Completo')).reset_index()

# Filter cities that have an active sentinel network
dsei_cities_ms = dsei2[dsei2.ones.notna()]

# Identify DSEIs without sentinel coverage
dsei_sem_sent = dsei2[~dsei2.ones.notna()]

# --- Optimization ---

# Join mobility data with DSEI data based on the city code
dsei2 = dsei.set_index('idm_ioibge').join(df_mob.set_index('cod_ibge_muni')).reset_index()

# Fill missing values in mobility coverage with 0
dsei2['City coverage(%)'] = dsei2['City coverage(%)'].fillna(0)

# Sort cities based on their mobility coverage in ascending order
dsei2 = dsei2.sort_values(by='City coverage(%)')

# Group by DSEI region and create a dictionary where keys are DSEI names and values are lists of city IDs
dsei_regions = dsei2[dsei2.NumDSEI == 1].groupby("NomeDSEI")["idm_ioibge"].apply(list).to_dict()

# Create a dictionary of city coverage percentages for each city
mobility_coverage = dsei2.groupby("idm_ioibge")["City coverage(%)"].apply(float).to_dict()

# List of all cities in the dataset
cities = dsei2.idm_ioibge.to_list()

# Set the number of cities to select in the optimization problem
N = 199  # Number of cities to select for the sentinel network

# --- Define Optimization Problem ---

# Create a linear programming problem with maximization objective
model = LpProblem(name="Sentinel_Network_Optimization", sense=LpMaximize)

# Decision variables: Binary selection for each city (1 if selected, 0 if not)
x = {city: LpVariable(name=f"x_{city}", cat="Binary") for city in cities}

# Objective function: Maximize total mobility coverage for the selected cities
model += lpSum(mobility_coverage[city] * x[city] for city in cities), "Maximize_Mobility_Coverage"

# Constraint 1: Select exactly N cities (no more, no less)
model += lpSum(x[city] for city in cities) == N, "Select_N_Cities"

# Constraint 2: Ensure at least one city from each DSEI region is selected
for dsei, city_list in dsei_regions.items():
    model += lpSum(x[city] for city in city_list) >= 1, f"Ensure_Coverage_{dsei}"

# Solve the optimization problem
model.solve()

# Extract the selected cities from the optimization result
selected_cities = [city for city in cities if x[city].value() == 1]

# --- Output Data ---

# Filter the final dataset to show the selected cities and relevant columns
final_result = dsei2[dsei2.idm_ioibge.isin(selected_cities)][['idm_ioibge', 'NomeDSEI', 'StatusDSEI', 
                                                           'NumDSEI', 'muni_name',
                                                           'uf_muni', 'City coverage(%)', 
                                                           'Number of paths covered',
                                                           'MoH sentinel']]


# --- Descriptive Results Summary ---

# Number of cities in DSEIs covered by Brazil's flu sentinel network
covered_cities_count = len(dsei_cities_ms[dsei_cities_ms.NumDSEI == 1])
print(f"Number of cities in DSEIs covered by the Brazil's flu sentinel network: {covered_cities_count}")

# Number of DSEIs with a sentinel network
dsei_with_sentinel_count = len(dsei_cities_ms[dsei_cities_ms.NumDSEI == 1].groupby(['NomeDSEI'])['ones'].sum().reset_index())
print(f"Number of DSEIs with a sentinel network: {dsei_with_sentinel_count}")

# Extract lists of DSEIs covered and uncovered by the sentinel network
covered_dseis = list(dsei_cities_ms[dsei_cities_ms.NumDSEI == 1].groupby(['NomeDSEI'])['ones'].sum().reset_index().NomeDSEI)
uncovered_dseis = list(dsei_sem_sent[dsei_sem_sent.NumDSEI == 1].groupby(['NomeDSEI'])['ones'].sum().reset_index().NomeDSEI)

# List DSEIs uncovered by the Brazilian sentinel network
uncovered_dseis_set = set(uncovered_dseis) - set(covered_dseis)
print(f"List of DSEIs uncovered by the Brazilian sentinel network: {uncovered_dseis_set}")

# Estimated population of uncovered DSEIs
uncovered_dseis_population = dsei_sem_sent[dsei_sem_sent.NomeDSEI.isin(uncovered_dseis_set)].populacao.sum()
print(f"Estimated population of uncovered DSEIs: {uncovered_dseis_population}")

# Estimated area of uncovered DSEIs
uncovered_dseis_area = dsei_sem_sent[dsei_sem_sent.NomeDSEI.isin(uncovered_dseis_set)].Area_MUN.sum()
print(f"Estimated area of uncovered DSEIs: {uncovered_dseis_area}")

# Total number of paths covered by the Brazilian sentinel network
paths_covered_by_sent = df_mob[df_mob['MoH sentinel'] == 1]['Number of paths covered'].sum()
print(f"Paths covered by the Brazilian sentinel network: {paths_covered_by_sent}")

# Total mobility coverage percentage by the Brazilian sentinel network
mobility_coverage_by_sent = df_mob[df_mob['MoH sentinel'] == 1]['City coverage(%)'].sum()
print(f"Mobility coverage by the Brazilian sentinel network: {mobility_coverage_by_sent}%")

# Total cities that are part of DSEI regions in the new optimized sentinel network
optimized_cities_count = len(final_result[final_result.NumDSEI != 0])
print(f"Total number of cities that are part of DSEI regions in the new optimized sentinel network: {optimized_cities_count}")

# Total mobility coverage in the new sentinel network
optimized_network_mobility = final_result['City coverage(%)'].sum()
print(f"Mobility coverage of the new sentinel network: {optimized_network_mobility}%")

# Total number of paths covered in the new optimized sentinel network
optimized_network_paths = final_result['Number of paths covered'].sum()
print(f"Number of paths covered in the new sentinel network: {optimized_network_paths}")

# Total number of paths in the initial sentinel network
total_paths = df_mob['Number of paths covered'].sum()
print(f"Total number of paths covered in the initial network: {total_paths}")
