import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import os

from pathlib import Path

# Paso 1: Leer los archivos CSV
output_dir='output'
mod_path = Path(__file__).parent
output_dir = mod_path / output_dir
df_episodios = pd.read_csv(os.path.join(output_dir, 'clean_order_data.csv')) 
df_vehiculos = pd.read_csv(os.path.join(output_dir, 'processed_vehicle_data.csv')) 

# Paso 2 y 3: Combinar los dataframes y añadir la columna de tipo de carga
df_combinado = pd.merge(df_episodios, df_vehiculos[['tag_id', 'charge_speed']], on='tag_id', how='left')
def parse_datetime(date_string):
            try:
                return pd.to_datetime(date_string, format='%Y-%m-%d %H:%M:%S.%f%z')
            except ValueError:
                return pd.to_datetime(date_string, format='%Y-%m-%d %H:%M:%S%z')
        

df_combinado['started_at'] = df_combinado['started_at'].apply(parse_datetime)
df_combinado = df_combinado.sort_values('started_at', ascending=True)


# Paso 4: Calcular los tiempos entre cargas para cada tipo de vehículo

d=[]
for i in df_combinado['charge_speed'].unique():
    datos = df_combinado[df_combinado['charge_speed'] == i]
    datos['time_between_charges'] = datos['started_at'].diff().dt.total_seconds() / 3600
    datos = datos[datos['time_between_charges']>1]
    d.append(datos)
df_combinado = pd.concat(d)
df_combinado = df_combinado.dropna()
# Guardar el resultado en un nuevo archivo CSV
df_combinado.to_csv('episodios_carga_con_tipo_ordenado.csv', index=False)

# Paso 5: Análisis de bondad de ajuste a la distribución de Poisson
def test_multiple_distributions(data, charge_speed, significance_level=0.05):
    """
    Perform a statistical analysis to test the goodness of fit of different probability distributions to the given data.

    Args:
        data (array-like): The data to be analyzed.
        significance_level (float, optional): The significance level for the Kolmogorov-Smirnov test. Defaults to 0.05.

    Returns:
        dict: A dictionary containing the results of the analysis. The keys are the names of the tested distributions,
              and the values are dictionaries with the following keys:
              - 'params' (tuple): The parameters of the fitted distribution.
              - 'ks_statistic' (float): The Kolmogorov-Smirnov statistic.
              - 'p_value' (float): The p-value of the test.

    This function performs a statistical analysis to test the goodness of fit of different probability distributions to the
    given data. It first sorts the data and calculates the empirical cumulative distribution function (CDF). Then, it defines
    the distributions to be tested and iterates over them. For each distribution, it fits the data using the appropriate
    method and calculates the theoretical CDF. It then performs the Kolmogorov-Smirnov test to compare the empirical CDF
    with the theoretical CDF. The results are stored in a dictionary, along with the parameters of the fitted distribution and
    the p-value of the test. Finally, it plots the empirical CDF and the theoretical CDFs and displays the plot.

    Note:
        The supported distributions are Poisson, Exponential, Gamma, Log-normal, and Weibull.

    """
    # Ordenar los datos y calcular la CDF empírica
    sorted_data = np.sort(data)
    ecdf = np.arange(1, len(data) + 1) / len(data)
    
    # Definir las distribuciones a probar
    distributions = {
        'Poisson': stats.poisson,
        'Exponential': stats.expon,
        'Gamma': stats.gamma,
        'Log-normal': stats.lognorm,
        'Weibull': stats.weibull_min
    }
    
    results = {}
    
    fig, axs = plt.subplots(3, 2, figsize=(15, 20))
    axs = axs.ravel()
    # Ajustar el espacio entre subplots
    plt.subplots_adjust(top= 0.75, bottom=0.12, left=0.10, right=0.90, hspace=0.4, wspace=0.3)
    
    for idx, (name, distribution) in enumerate(distributions.items()):
        ax = axs[idx]
        # Ajustar la distribución a los datos
        if name == 'Poisson':
            params = (np.mean(data),)  # lambda es la media para Poisson
        elif name == 'Log-normal':
            params = stats.lognorm.fit(data, floc=0)
        else:
            params = distribution.fit(data)
        
        # Calcular la CDF teórica
        if name == 'Poisson':
            cdf_theo = distribution.cdf(np.floor(sorted_data), *params)
        else:
            cdf_theo = distribution.cdf(sorted_data, *params)
        
        # Realizar la prueba de Kolmogorov-Smirnov
        if name == 'Poisson':
            ks_statistic, p_value = stats.kstest(data, lambda x: distribution.cdf(np.floor(x), *params))
        else:
            ks_statistic, p_value = stats.kstest(data, distribution.cdf, args=params)
        
        # Guardar resultados
        results[name] = {
            'params': params,
            'ks_statistic': ks_statistic,
            'p_value': p_value
        }
        
         # Dibujar la ECDF
        ax.step(sorted_data, ecdf, where='post', label='ECDF (Empírica)', 
                linewidth=2, color='blue', alpha=0.7)
        ax.fill_between(sorted_data, 0, ecdf, step='post', alpha=0.3, color='blue')
        
        # Dibujar la CDF teórica
        ax.plot(sorted_data, cdf_theo, label=f'{name} CDF', color='red', linewidth=2)
        
        ax.set_xlabel('Tiempo entre llegadas')
        ax.set_ylabel('Probabilidad acumulada')
        ax.set_title(f'CDF Empírica vs {name}', pad = 20)
        ax.grid(True, alpha=0.3)
        
        # Mover la leyenda a la esquina superior izquierda
        ax.legend(loc='upper left')
        
        # Añadir texto con los resultados del test en la esquina inferior derecha
        ax.text(0.95, 0.05, f'KS statistic: {ks_statistic:.4f}\np-value: {p_value:.4f}', 
                transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
   # Eliminar el subplot extra
    axs[-1].axis('off')
    fig.suptitle(f'Análisis de distribuciones de probabilidad para la velocidad {charge_speed}', y=1.05, fontsize=16)
    plt.tight_layout()

    plt.savefig(output_dir / f'distribucion_{charge_speed}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

def calcular_intervalo_confianza(data, distribution, alpha=0.05, hours=24):
    """
    Calculates the confidence interval for a given distribution.

    Args:
        data (array-like): The data to fit the distribution to.
        distribution (scipy.stats.rv_continuous): The distribution to fit the data to.
        alpha (float, optional): The significance level of the confidence interval. Defaults to 0.05.
        hours (float, optional): The number of hours to estimate the average arrival time. Defaults to 24.

    Returns:
        tuple: A tuple containing the average arrival time (lambda_24h) and the lower and upper bounds of the confidence interval.

    Raises:
        ValueError: If the distribution is not supported.
    """
    # Ajustar la distribución a los datos
    params = distribution.fit(data)
    
    # Calcular el tiempo medio entre llegadas
    mean_time = distribution.mean(*params)
    
    # Estimar el número medio de llegadas en 24 horas
    lambda_24h = hours / mean_time
    
    if distribution == stats.gamma:
        # Para la distribución Gamma
        lower = stats.gamma.ppf(alpha/2, a=lambda_24h, scale=1)
        upper = stats.gamma.ppf(1-alpha/2, a=lambda_24h, scale=1)
    elif distribution == stats.weibull_min:
        # Para la distribución Weibull
        c = params[0]  # parámetro de forma
        scale = params[2]  # parámetro de escala
        lambda_weibull = (hours / scale) ** c
        lower = stats.weibull_min.ppf(alpha/2, c, scale=lambda_weibull**(1/c))
        upper = stats.weibull_min.ppf(1-alpha/2, c, scale=lambda_weibull**(1/c))
    else:
        raise ValueError("Distribución no soportada")
    
    return lambda_24h, (lower, upper)



# Realizar el análisis para cada tipo de carga
significance_level = 0.05

fichero = os.path.join(output_dir, 'analisis_estadistico.txt') 
with open(fichero, 'w', encoding='utf-8') as f:
    for tipo in df_combinado['charge_speed'].unique():
        datos_tipo = df_combinado[df_combinado['charge_speed'] == tipo]['time_between_charges']
        print(f"\nTipo de carga: {tipo}")
        
        results = test_multiple_distributions(datos_tipo, tipo)
        
        f.write("Análisis de distribuciones:\n")
        f.write("-" * 30 + "\n")
        for name, result in results.items():
            f.write(f"\nDistribución: {name}\n")
            if name == 'Poisson':
                f.write(f"Lambda estimado: {result['params'][0]:.4f}\n")
            else:
                f.write(f"Parámetros estimados: {result['params']}\n")
            f.write(f"Estadístico KS: {result['ks_statistic']:.4f}\n")
            f.write(f"Valor p: {result['p_value']:.4f}\n")
            
            if result['p_value'] > 0.05:
                f.write("No se puede rechazar la hipótesis de esta distribución\n")
            else:
                f.write("Se rechaza la hipótesis de esta distribución\n")
        
        f.write("\nIntervalos de confianza para 24 horas:\n")
        f.write("-" * 40 + "\n")
        for dist_name, distribution in [("Gamma", stats.gamma), ("Weibull", stats.weibull_min)]:
            mean, (lower, upper) = calcular_intervalo_confianza(datos_tipo, distribution)
            
            f.write(f"\nDistribución: {dist_name}\n")
            f.write(f"Número medio estimado de llegadas en 24 horas: {mean:.2f}\n")
            f.write(f"Intervalo de confianza del 95%: ({lower:.2f}, {upper:.2f})\n")
        
        f.write("\n" + "=" * 50 + "\n")

print("Análisis completado. Los resultados se han guardado en el directorio 'resultados'.")

