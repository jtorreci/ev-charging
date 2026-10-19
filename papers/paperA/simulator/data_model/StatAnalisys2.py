import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import json
import os

from pathlib import Path


import holidays


class Simulation_stochastic_analysis:
    def __init__(self, output_dir='output'):
        """
        Constructor for the Simulation_stochastic_analysis class.

        Parameters:
            output_dir (str): Directory where the simulation results will be saved.
            Default is './Data/output'.
        """
        mod_path = Path(__file__).parent
        self.output_dir = mod_path / output_dir
        # self.output_dir = output_dir
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)

    def load_and_process_data(self, schedule_params = None):
        """
        Reads and processes charging episodes and vehicle data.
        Reads the CSV files 'clean_order_data.csv' and 'processed_vehicle_data.csv',
        combines them, and adds the 'charge_speed' column from the vehicles.
        Then, it calculates the time between charges for each type of vehicle,
        filtering out values less than 1 hour.
        Finally, it saves the result in a new CSV file called
        'episodios_carga_con_tipo_ordenado.csv' in the directory specified
        in the constructor.
        Returns the processed dataframe.

        """
        # Paso 1: Leer los archivos CSV
        clean_data_path = os.path.join(self.output_dir, 'clean_order_data.csv')
        processed_data_path = os.path.join(self.output_dir, 'processed_vehicle_data.csv')
        
        # Verificar existencia de archivos de entrada
        if not os.path.exists(clean_data_path):
            raise FileNotFoundError(
                f"Archivo de datos limpios no encontrado: {clean_data_path}\n"
                f"Debe ejecutar el preprocesamiento de datos primero."
            )
        if not os.path.exists(processed_data_path):
            raise FileNotFoundError(
                f"Archivo de datos procesados no encontrado: {processed_data_path}\n"
                f"Debe ejecutar el preprocesamiento de datos primero."
            )
        
        df_episodios = pd.read_csv(clean_data_path)
        df_vehiculos = pd.read_csv(processed_data_path)

        # Paso 2 y 3: Combinar los dataframes y anadir la columna de tipo de carga
        df_combinado = pd.merge(df_episodios, df_vehiculos[['tag_id', 'charge_speed']], on='tag_id', how='left')
        # Convertir 'started_at' a datetime, manejando ambos formatos
        def parse_datetime(date_string):
            try:
                return pd.to_datetime(date_string, format='%Y-%m-%d %H:%M:%S.%f%z')
            except ValueError:
                return pd.to_datetime(date_string, format='%Y-%m-%d %H:%M:%S%z')
        

        df_combinado['started_at'] = df_combinado['started_at'].apply(parse_datetime)
        df_combinado = df_combinado.sort_values('started_at', ascending=True)
        es_holidays = holidays.Spain()
        
        try:
            df_combinado['dia_semana'] = df_combinado['started_at'].dt.day_name(locale='es_ES')
        except:
            # Fallback if locale is not available
            english_to_spanish = {
                'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
                'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado',
                'Sunday': 'Domingo'
            }
            df_combinado['dia_semana'] = df_combinado['started_at'].dt.day_name().map(english_to_spanish)

        def is_holiday(date):
            if date.weekday() >= 5 or date in es_holidays:
                return "Festivo"
            return "Laborable"
        
        df_combinado['es_festivo'] = df_combinado['started_at'].dt.date.apply(is_holiday)

        # Paso 4: Calcular los tiempos entre cargas para cada tipo de vehículo
        d = []
        for i in df_combinado['charge_speed'].unique():
            datos = df_combinado[df_combinado['charge_speed'] == i].copy()  # Crear una copia explícita
            datos['time_between_charges'] = datos['started_at'].diff().dt.total_seconds() / 3600
            datos = datos[datos['time_between_charges'] > 1]
            d.append(datos)
        df_combinado = pd.concat(d)
        df_combinado = df_combinado.dropna()
        if schedule_params is not None:
            if schedule_params['HOLIDAYS']:
                df_combinado = df_combinado[df_combinado['es_festivo'] == 'Laborable']
                print (df_combinado.head())
            if schedule_params['WEEKDAYS']:
                df_combinado = df_combinado[df_combinado['dia_semana'].isin(schedule_params['WEEKDAYS'])]
                print (df_combinado.head())
        # Guardar el resultado en un nuevo archivo CSV
        df_combinado.to_csv(os.path.join(self.output_dir, 'episodios_carga_con_tipo_ordenado.csv'), index=False)

        return df_combinado

    def test_multiple_distributions(self, data, charge_speed, significance_level=0.05, skip_plots=False):
        """
        Perform a statistical analysis to test the goodness of fit of different probability distributions to the given data.

        Args:
            data (array-like): The data to be analyzed.
            charge_speed (str): The charge speed type associated with the data.
            significance_level (float, optional): The significance level for the Kolmogorov-Smirnov test. Defaults to 0.05.
            skip_plots (bool, optional): If True, skip generating plots. Defaults to False.

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
        sorted_data = np.sort(data)
        ecdf = np.arange(1, len(data) + 1) / len(data)

        distributions = {
            'Poisson': stats.poisson,
            'Exponential': stats.expon,
            'Gamma': stats.gamma,
            'Log-normal': stats.lognorm,
            'Weibull': stats.weibull_min
        }

        results = {}

        if not skip_plots:
            fig, axs = plt.subplots(3, 2, figsize=(15, 20))
            axs = axs.ravel()
            plt.subplots_adjust(top=0.75, bottom=0.12, left=0.10, right=0.90, hspace=0.4, wspace=0.3)

        for idx, (name, distribution) in enumerate(distributions.items()):
            if not skip_plots:
                ax = axs[idx]
            
            if name == 'Poisson':
                params = (np.mean(data),)
            elif name == 'Log-normal':
                params = stats.lognorm.fit(data, floc=0)
            else:
                params = distribution.fit(data)
            
            if name == 'Poisson':
                cdf_theo = distribution.cdf(np.floor(sorted_data), *params)
                ks_statistic, p_value = stats.kstest(data, lambda x: distribution.cdf(np.floor(x), *params))
            else:
                cdf_theo = distribution.cdf(sorted_data, *params)
                ks_statistic, p_value = stats.kstest(data, distribution.cdf, args=params)
            
            results[name] = {
                'params': params,
                'ks_statistic': ks_statistic,
                'p_value': p_value
            }

            if not skip_plots:
                ax.step(sorted_data, ecdf, where='post', label='ECDF (Empírica)',
                        linewidth=2, color='blue', alpha=0.7)
                ax.fill_between(sorted_data, 0, ecdf, step='post', alpha=0.3, color='blue')

                ax.plot(sorted_data, cdf_theo, label=f'{name} CDF', color='red', linewidth=2)

                ax.set_xlabel('Tiempo entre llegadas')
                ax.set_ylabel('Probabilidad acumulada')
                ax.set_title(f'CDF Empírica vs {name}', pad=20)
                ax.grid(True, alpha=0.3)

                ax.legend(loc='upper left')

                ax.text(0.95, 0.05, f'KS statistic: {ks_statistic:.4f}\np-value: {p_value:.4f}',
                        transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        if not skip_plots:
            axs[-1].axis('off')
            fig.suptitle(f'Análisis de distribuciones de probabilidad para la velocidad {charge_speed}', y=1.05, fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'distribucion_{charge_speed}.png'), dpi=300, bbox_inches='tight')
            plt.close()

        return results

    def calcular_intervalo_confianza(self, data, distribution, alpha=0.05, hours=24):
        """
        Calcula el intervalo de confianza para un conjunto de datos dado y una distribución de probabilidad.

        Args:
            data (array-like): Los datos a analizar.
            distribution (scipy.stats.rv_continuous): La distribución de probabilidad a utilizar.
            alpha (float, optional): El nivel de significación. Defaults to 0.05.
            hours (float, optional): El número de horas a considerar. Defaults to 24.

        Returns:
            tuple: Un tuple conteniendo el número medio estimado de llegadas en 24 horas y el intervalo de confianza del 95%.

        Raises:
            ValueError: Si la distribución no es soportada.
        """
        params = distribution.fit(data)
        mean_time = distribution.mean(*params)
        lambda_24h = hours / mean_time
        
        if distribution == stats.gamma:
            lower = stats.gamma.ppf(alpha/2, a=lambda_24h, scale=1)
            upper = stats.gamma.ppf(1-alpha/2, a=lambda_24h, scale=1)
        elif distribution == stats.weibull_min:
            c = params[0]
            scale = params[2]
            lambda_weibull = (hours / scale) ** c
            lower = stats.weibull_min.ppf(alpha/2, c, scale=lambda_weibull**(1/c))
            upper = stats.weibull_min.ppf(1-alpha/2, c, scale=lambda_weibull**(1/c))
        else:
            raise ValueError("Distribución no soportada")
        
        return lambda_24h, (lower, upper)

    def distribution_params_exist(self, charge_types=None):
        """
        Check if distribution parameter JSON files already exist.
        
        :param charge_types: List of charge types to check (e.g., ['Slow', 'Normal', 'Fast']).
                            If None, checks for all standard types.
        :return: True if all parameter files exist, False otherwise.
        """
        if charge_types is None:
            charge_types = ['Slow', 'Normal', 'Fast']
        
        for ct in charge_types:
            json_path = os.path.join(self.output_dir, f'distribution_params_{ct}.json')
            if not os.path.exists(json_path):
                return False
        return True
    
    def run_analysis(self, schedule_params = None, skip_plots=False):
        """
        Ejecuta el análisis estadístico de los datos y guarda los resultados en un archivo
        de texto y en un archivo JSON.

        Primero, se cargan y procesan los datos. Luego, se itera sobre cada tipo de carga
        y se realizan las siguientes operaciones para cada una:

        1. Se prueban varias distribuciones de probabilidad para ajustar a los datos.
        2. Se calculan los parámetros de ajuste para cada distribución.
        3. Se calcula el estadístico de Kolmogorov-Smirnov y su valor p.
        4. Se escribe el resultado del ajuste en un archivo de texto.
        5. Se guarda el resultado del ajuste en un archivo JSON.

        Finalmente, se escribe un mensaje indicando que el análisis ha sido completado
        y se han guardado los resultados en el directorio 'output'.
        
        :param skip_plots: If True, skip generating PNG plots (useful for batch/parallel runs).
        """
        df_combinado = self.load_and_process_data(schedule_params)
        significance_level = 0.05

        with open(os.path.join(self.output_dir, 'analisis_estadistico.txt'), 'w', encoding='utf-8') as f:
            for tipo in df_combinado['charge_speed'].unique():
                datos_tipo = df_combinado[df_combinado['charge_speed'] == tipo]['time_between_charges']
                energy_data = df_combinado[df_combinado['charge_speed'] == tipo]['energy']
                print(f"\nTipo de carga: {tipo}")

                results = self.test_multiple_distributions(datos_tipo, tipo, skip_plots=skip_plots)
                
                # Guardar los resultados en un archivo JSON
                json_results = {
                    'distribution_params': results,
                    'energy_stats': {
                        'mean': np.mean(energy_data),
                        'std': np.std(energy_data)
                    }
                }
                with open(os.path.join(self.output_dir, f'distribution_params_{tipo}.json'), 'w') as json_file:
                    json.dump(json_results, json_file, indent=4)
                
                f.write(f"\nTipo de carga: {tipo}\n")
                f.write("=" * 50 + "\n\n")
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
                    
                    if result['p_value'] > significance_level:
                        f.write("No se puede rechazar la hipótesis de esta distribución\n")
                    else:
                        f.write("Se rechaza la hipótesis de esta distribución\n")
                
                f.write("\nIntervalos de confianza para 24 horas:\n")
                f.write("-" * 40 + "\n")
                for dist_name, distribution in [("Gamma", stats.gamma), ("Weibull", stats.weibull_min)]:
                    mean, (lower, upper) = self.calcular_intervalo_confianza(datos_tipo, distribution)
                    
                    f.write(f"\nDistribución: {dist_name}\n")
                    f.write(f"Número medio estimado de llegadas en 24 horas: {mean:.2f}\n")
                    f.write(f"Intervalo de confianza del 95%: ({lower:.2f}, {upper:.2f})\n")
                
                f.write("\n" + "=" * 50 + "\n")

        print("Análisis completado. Los resultados se han guardado en el directorio 'output'.")

    def generate_random_series(self, tipo_carga, num_days, Debug=False):
        """
        Genera una serie de número de vehículos y energías para cada día, 
        utilizando la distribución de probabilidad que mejor se ajusta a los datos.

        Args:
            tipo_carga (str): El tipo de carga (Rapido, Lento, etc.).
            num_days (int): El número de días a simular.
            Debug (bool, optional): Si es True, se imprimen los primeros 5 días de la serie. Defaults to False.

        Returns:
            tuple: Un tuple conteniendo la serie de número de vehículos y la serie de energías.
        """
        file_path = os.path.join(self.output_dir, f'distribution_params_{tipo_carga}.json')
        with open(file_path, 'r') as f:
            params = json.load(f)
        
        best_dist = max(params['distribution_params'], key=lambda x: params['distribution_params'][x]['p_value'])
        best_params = params['distribution_params'][best_dist]['params']
        
        # Generar serie de número de vehículos por día
        if best_dist == 'Poisson':
            lambda_param = best_params[0]
            vehicle_series = np.random.poisson(24 / lambda_param, num_days)
        elif best_dist == 'Exponential':
            scale = best_params[0]
            vehicle_series = np.random.exponential(24 / scale, num_days)
        elif best_dist == 'Gamma':
            a, _, scale = best_params
            adjusted_scale = 24 / (a * scale)
            vehicle_series = np.random.gamma(a, adjusted_scale, num_days)
        elif best_dist == 'Log-normal':
            s, _, scale = best_params
            vehicle_series = np.random.lognormal(np.log(24 / scale), s, num_days)
        elif best_dist == 'Weibull':
            c, _, scale = best_params
            adjusted_scale = 24 / scale
            vehicle_series = np.random.weibull(c, num_days) * adjusted_scale
        
        # Redondear y convertir a enteros
        vehicle_series = np.round(vehicle_series).astype(int)

        # Filtrar y rellenar la serie de vehículos
        vehicle_series = self.filter_and_fill_series(vehicle_series)
        
        # Generar serie de energías como lista de listas
        # Use truncated normal (clamp to min 500 Wh) to avoid negative energies
        # from the Normal distribution when std is large relative to mean.
        energy_mean = params['energy_stats']['mean']
        energy_std = params['energy_stats']['std']
        MIN_ENERGY_WH = 500  # Minimum physically meaningful energy demand
        energy_series = [
            np.maximum(MIN_ENERGY_WH,
                       np.random.normal(energy_mean, energy_std,
                                        int(num_vehicles))).tolist()
            if num_vehicles > 0 else []
            for num_vehicles in vehicle_series
        ]

        if Debug:
            print(f"Primeros 5 días - Número de vehículos: {vehicle_series[:5]}")
            print(f"Primeros 5 días - Energías: {energy_series[:5]}")
            
        return vehicle_series.tolist(), energy_series
    
    def filter_and_fill_series(self, series):
        """
        Filters a series to include only values between the 10th and 90th percentiles,
        then fills the original series with the mean of these filtered values.

        Args:
            series (np.ndarray): The input series of values to filter and fill.

        Returns:
            np.ndarray: A new series where values within the original 10th to 90th percentile 
            range are retained, and all other values are replaced with the mean of these filtered values.
        """
        if len(series) == 0:
            return series

        # Calcular percentiles
        p10 = np.percentile(series, 10)
        p90 = np.percentile(series, 90)
        
        # Filtrar valores
        filtered_indices = (series >= p10) & (series <= p90)
        filtered_series = series[filtered_indices]
        
        # Calcular la media de los valores filtrados
        mean_value = np.mean(filtered_series)
        
        # Crear una nueva serie con la misma longitud que la original
        new_series = np.full_like(series, fill_value=int(round(mean_value)))
        
        # Colocar los valores filtrados en sus posiciones originales
        new_series[filtered_indices] = filtered_series
        
        return new_series

    def generate_all_random_series(self, num_days):
        """
        Generates a dictionary of random series for all charge types in the data. The output dictionary
        has the charge type as key and a dictionary with two keys: 'vehicles' and 'energies'. The 'vehicles'
        key corresponds to a list of random integers representing the number of vehicles for each day in the
        simulation. The 'energies' key corresponds to a list of random integers representing the total energy
        required by the vehicles for each day in the simulation.

        Parameters
        ----------
        num_days : int
            The number of days to simulate.

        Returns
        -------
        dict
            A dictionary of random series for all charge types in the data.
        """
        df_combinado = self.load_and_process_data()
        tipos_carga = df_combinado['charge_speed'].unique()
        random_series = {}
        
        for tipo in tipos_carga:
            vehicle_series, energy_series = self.generate_random_series(tipo, num_days)
            random_series[tipo] = {
                'vehicles': vehicle_series,
                'energies': energy_series
            }
        
        return random_series

if __name__=="__main__":
    # Crear una instancia de la clase
    stochastic_analysis = Simulation_stochastic_analysis(output_dir='./output')

    # Ejecutar el análisis y guardar los parámetros
    stochastic_analysis.run_analysis()

    # Generar series aleatorias
    num_samples = 1000  # O el número que desees
    random_series = stochastic_analysis.generate_all_random_series(num_samples)

    # Ahora puedes usar estas series aleatorias en tu simulación
    for tipo in random_series.keys():
        print(f"Tipo de carga: {tipo}")
        print(f"Número de vehículos (primeros 10): {random_series[tipo]['vehicles'][:10]}")
        print(f"Energías (primeros 10): {random_series[tipo]['energies'][:10]}")
        print()

