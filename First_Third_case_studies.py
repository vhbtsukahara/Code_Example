import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import pymc as pm
import numpy as np
import arviz as az
from sklearn.preprocessing import LabelEncoder
from lifelines import CoxPHFitter

# Load the spreadsheets (adjust paths as necessary)
ventilador_inv = pd.read_excel('ventilador_inv.xlsx')
ventilador_hist = pd.read_excel('ventilador_hist.xlsx')

# Check if there is a match in 'Ebserh' or 'Serie' between the two spreadsheets
ventilador_inv['Presente_Hist'] = ventilador_inv.apply(
    lambda row: (row['Ebserh'] in ventilador_hist['Ebserh'].values) or (row['Serie'] in ventilador_hist['Serie'].values),
    axis=1
)

# Display the result
#print(ventilador_inv[['Ebserh', 'Serie', 'Presente_Hist']])

### Mean Time of Corrective Maintenance / Equipment - for each Manufacturer ###

# Get the current date
data_atual = datetime.now()

# Calculate the age in years and add it to the new column 'Idade'
ventilador_inv['Idade'] = ventilador_inv['Aquisição'].apply(lambda x: (data_atual - x).days // 365)

# Extract the year from the 'Aquisição' column and add it to the new column 'Ano'
ventilador_inv['Ano'] = ventilador_inv['Aquisição'].dt.year

# Filter the records of 'Corrective Maintenance' in the 'ventilador_hist' spreadsheet
manutencao_corretiva = ventilador_hist[ventilador_hist['Tipo'] == 'Manutenção Corretiva']

# Count the number of occurrences of corrective maintenance per equipment
contagem_manutencao = manutencao_corretiva.groupby(['Ebserh', 'Serie']).size().reset_index(name='Numero')

# Get the most recent corrective maintenance date per equipment
ultimo_momento = manutencao_corretiva.groupby(['Ebserh', 'Serie'])['Morte'].max().reset_index(name='Momento')

# Merge the maintenance count and date information with the 'ventilador_inv' spreadsheet
ventilador_inv = ventilador_inv.merge(contagem_manutencao, on=['Ebserh', 'Serie'], how='left')
ventilador_inv = ventilador_inv.merge(ultimo_momento, on=['Ebserh', 'Serie'], how='left')

# Create the 'Falha' column, indicating True if there is a corrective maintenance record, and False otherwise
ventilador_inv['Falha'] = ventilador_inv['Numero'].notna()

# Get the duration of corrective maintenance for each equipment
duracao_manutencao = manutencao_corretiva.groupby(['Ebserh', 'Serie'])['Duração'].sum().reset_index(name='Duração_Total')

# Merge the total corrective maintenance duration with the 'ventilador_inv' spreadsheet
ventilador_inv = ventilador_inv.merge(duracao_manutencao, on=['Ebserh', 'Serie'], how='left')

ventilador_inv['Marca'] = ventilador_inv['Marca'].replace('NELLCOR PURITAN BENNETT', 'PURITAN BENNETT')

# Filter the records of 'Scheduled Maintenance' in the 'ventilador_hist' spreadsheet
manutencao_programada = ventilador_hist[ventilador_hist['Tipo'] == 'Manutenção Programada']

# Create the columns 'Preventiva' and 'Preventiva_data' in the 'ventilador_inv_final' spreadsheet
ventilador_inv['Preventiva'] = False
ventilador_inv['Preventiva_data'] = None

# Check the existence of scheduled maintenance for each equipment in the 'ventilador_inv_final' spreadsheet
for index, row in ventilador_inv.iterrows():
    ebserh = row['Ebserh']
    serie = row['Serie']

    # Check in 'manutencao_programada' if there is equipment with the same 'Ebserh' or 'Serie'
    match = manutencao_programada[
        (manutencao_programada['Ebserh'] == ebserh) | (manutencao_programada['Serie'] == serie)
    ]

    if not match.empty:
        ventilador_inv.at[index, 'Preventiva'] = True
        ventilador_inv.at[index, 'Preventiva_data'] = match['Morte'].values[0]

# Filter only the records with failure
manutencao_corretiva = ventilador_inv[ventilador_inv['Falha'] == True]

# Recalculate average age of equipment by brand
idade_media_marca = manutencao_corretiva.groupby('Marca')['Idade'].mean()

# Calculate the count of equipment by brand to normalize maintenance time
quantitativo_marca = ventilador_inv['Marca'].value_counts()

# Recalculate average corrective maintenance time by brand and normalize by quantity
media_duracao_marca = manutencao_corretiva.groupby('Marca')['Duração_Total'].mean()
tempo_normalizado = media_duracao_marca / quantitativo_marca

# Combine average age and normalized average corrective maintenance time into a single dataframe
dados_combinados = pd.DataFrame({
    'Normalized Mean Corrective Maintenance Time (hours/device)': tempo_normalizado,
    'Average Age (years)': idade_media_marca
}).sort_values(by='Normalized Mean Corrective Maintenance Time (hours/device)', ascending=False)

# Filter the 10 most frequent brands in corrective maintenance
marcas_frequentes = manutencao_corretiva['Marca'].value_counts().head(10).index
dados_combinados_top_10 = dados_combinados.loc[marcas_frequentes]

# Sort the dataframe of the top 10 most frequent brands by normalized average maintenance time
dados_combinados_top_10 = dados_combinados_top_10.sort_values(by='Normalized Mean Corrective Maintenance Time (hours/device)', ascending=False)

# Plot the chart
fig, ax = plt.subplots(figsize=(12, 8))
dados_combinados_top_10['Normalized Mean Corrective Maintenance Time (hours/device)'].plot(kind='bar', ax=ax, color='#347eff')

# Add average age labels above the bars
for i, (tempo, idade) in enumerate(zip(dados_combinados_top_10['Normalized Mean Corrective Maintenance Time (hours/device)'], dados_combinados_top_10['Average Age (years)'])):
    ax.text(i, tempo + 0.5, f'{idade:.1f} years' if pd.notna(idade) else 'N/A', ha='center', va='bottom')

# Chart settings
plt.xlabel("Manufacturer")
plt.ylabel("Normalized Mean Time of Corrective Maintenance (hours/device)")
#plt.title("Comparison of Normalized Mean Corrective Maintenance Time and Average Age among the Top 10 Brands (Descending Order)")
plt.xticks(rotation=45)
plt.tight_layout()

# Show the chart
plt.savefig('brands.png', bbox_inches='tight')
plt.show()

### Cox Proportional Hazards code ###
# Definir as marcas que queremos ordenar e considerar como categorias no modelo
marca_ordenada = ['MINDRAY', 'DIXTAL', 'INTERMED', 'PURITAN BENNETT', 'DRAGER', 'MAQUET', 'LEISTUNG', 'GE HEALTHCARE', 'TECME', 'CAREFUSION']

# Definir o período de observação
start_date = pd.to_datetime("2022-01-01")
end_date = pd.to_datetime("2024-11-11")

# Filtrar os dados para as 10 marcas mais frequentes
top_10_brands = ventilador_inv['Marca'].value_counts().nlargest(10).index
filtered_df = ventilador_inv[ventilador_inv['Marca'].isin(top_10_brands)].copy()

# Ajustar as colunas de datas
filtered_df['Momento'] = pd.to_datetime(filtered_df['Momento'], errors='coerce')
filtered_df['Aquisição'] = pd.to_datetime(filtered_df['Aquisição'], errors='coerce')

# Calcular o tempo para falha ou censura
filtered_df['Tempo'] = (filtered_df['Momento'].fillna(end_date) - start_date).dt.days
filtered_df['Evento'] = filtered_df['Falha'].fillna(False).astype(int)

# Configurar a coluna 'Marca' como uma variável categórica com a ordem desejada
filtered_df['Marca'] = pd.Categorical(filtered_df['Marca'], categories=marca_ordenada, ordered=True)

# Converter a coluna 'Marca' em variáveis dummies com a primeira categoria como referência
df_cox = pd.get_dummies(filtered_df[['Tempo', 'Evento', 'Marca']], drop_first=True)

# Inicializar o modelo de Cox
cph = CoxPHFitter()
cph.fit(df_cox, duration_col='Tempo', event_col='Evento')

# Plotar o modelo ajustado de Cox para comparação entre marcas
cph.plot()

# Configurações do gráfico
#plt.title("Comparação Proporcional de Sobrevivência entre Marcas usando o Modelo de Cox")
plt.xlabel("Proportional risk of fail")
plt.ylabel("Manufacturer")
plt.grid(True)
plt.show()

### Bayesian Regression Code ###

# Data preparation
ventilador_inv_final['Hospital_encoded'] = LabelEncoder().fit_transform(ventilador_inv_final['Hospital'])
ventilador_inv_final['Marca_encoded'] = LabelEncoder().fit_transform(ventilador_inv_final['Marca'])
ventilador_inv_final['Contrato_encoded'] = ventilador_inv_final['Contrato'].apply(lambda x: 1 if x == 'SIM' else 0)
ventilador_inv_final['Preventiva_encoded'] = ventilador_inv_final['Preventiva'].astype(int)

# Select the variables
data = ventilador_inv_final[['Hospital_encoded', 'Marca_encoded', 'Contrato_encoded', 'Idade', 'Preventiva_encoded', 'Falha']].dropna()
X = data[['Hospital_encoded', 'Marca_encoded', 'Contrato_encoded', 'Idade', 'Preventiva_encoded']].values
y = data['Falha'].astype(int).values

# Bayesian model with pymc
with pm.Model() as model:
    # Priors for coefficients and intercept
    intercept = pm.Normal('Intercept', mu=0, sigma=10)
    coefs = pm.Normal('Coefs', mu=0, sigma=10, shape=X.shape[1])

    # Logistic regression formula
    logit_p = intercept + pm.math.dot(X, coefs)
    p = pm.Deterministic('p', pm.math.sigmoid(logit_p))

    # Likelihood
    observed_falha = pm.Bernoulli('observed_falha', p=p, observed=y)

    # Sampling the parameters
    trace = pm.sample(1000, chains=2, random_seed=42)

# Summary of traces for coefficient analysis
summary = pm.summary(trace, var_names=['Intercept', 'Coefs'])
print(summary)

# Check R-hat and ESS
rhat_values = az.rhat(trace)
ess_values = az.ess(trace)

print("R-hat values:", rhat_values)
print("Effective Sample Size (ESS):", ess_values)

# Plot trace and autocorrelation graphs to visually check convergence
az.plot_trace(trace)
az.plot_autocorr(trace)


