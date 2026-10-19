import pandas as pd
from dateutil import parser
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class DataProcessor:
    def __init__(self, file_path_pattern):
        """
        Constructor de la clase DataProcessor. 

        :param file_path_pattern: patron de ruta para buscar los archivos csv
        """
        self.file_path_pattern = file_path_pattern

    # Metodo para cargar todos los csv de datos disponibles y convertirlos en un unico conjunto de datos
    def load_and_concat_data(self):
        """Cargar y concatenar todos los archivos CSV en un único DataFrame"""
        all_files = glob.glob(self.file_path_pattern)
        data_list = []

        for file in all_files:
            df = pd.read_csv(file, sep=';')
            if 'started_at' in df.columns and 'stopped_at' in df.columns:
                data_list.append(df)
            else:
                print(f"El archivo {file} no contiene las columnas necesarias y será omitido.")

        if data_list:
            self.data = pd.concat(data_list, ignore_index=True)
        else:
            raise ValueError("Ninguno de los archivos CSV contiene las columnas necesarias.")
        
        self.data['started_at'] = self.data['started_at'].apply(self.parse_date)
        self.data['stopped_at'] = self.data['stopped_at'].apply(self.parse_date)

    # Metodo para convertir las fechas de los datos y poder utilizarlas sin errores
    def parse_date(self, date_str):
        """Convertir una cadena de fecha a objeto de fecha"""
        try:
            return parser.parse(date_str)
        except (ValueError, TypeError):
            return pd.NaT

    # Metodo para obtener los outliers de nuestros datos usando IQR
    def identify_outliers(self, column):
        """Identificar outliers usando el rango intercuartil (IQR)"""
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return self.data[column][(self.data[column] < lower_bound) | (self.data[column] > upper_bound)]


    
    def order_fill_data(self):
        """
        Llenar valores vacíos en las columnas 'battery_start' y 'battery_pct' basándose en la distribución observada.
        Luego, obtener la hora de llegada del vehículo y ordenar los datos por 'tag_id' y 'started_at'.
        """
        
        battery_start_non_null = self.data['battery_start'].dropna()
        battery_pct_non_null = self.data['battery_pct'].dropna()

        battery_start_filled = np.random.choice(battery_start_non_null, size=self.data['battery_start'].isnull().sum())
        battery_pct_filled = np.random.choice(battery_pct_non_null, size=self.data['battery_pct'].isnull().sum())

        self.data.loc[self.data['battery_start'].isnull(), 'battery_start'] = battery_start_filled
        self.data.loc[self.data['battery_pct'].isnull(), 'battery_pct'] = battery_pct_filled

        # Obtener hora a la que llega el vehículo
        self.data['arrival_hour'] = self.data['started_at'].dt.hour

        self.data = self.data.sort_values(['tag_id', 'started_at'])
        return self.data

    # Metodo para agrupar por coche calculando asi todas sus medias y caracteristicas
    def group_by_tagID(self,output_dir):
        # Calcular el tiempo entre cargas
        self.data['time_between_charges'] = self.data.groupby('tag_id')['started_at'].diff().dt.total_seconds() / 3600

        # Rellenar NaN con 0 en la columna 'time_between_charges'
        self.data['time_between_charges'].fillna(0)

        # Identificar outliers en 'time_between_charges' utilizando el método identify_outliers
        outliers = self.identify_outliers('time_between_charges')

        # Función para calcular la media excluyendo los valores de 0 y los outliers
        def mean_excluding_zeros_and_outliers(series):
            non_zero_values = series[series > 0]
            filtered_values = non_zero_values[~non_zero_values.isin(outliers)]
            return filtered_values.mean() if len(filtered_values) > 0 else 0

        # Función para calcular la probabilidad de no venir en la fecha común
        def calculate_outlier_probability(series):
            non_zero_values = series[series > 0]
            filtered_values = non_zero_values[~non_zero_values.isin(outliers)]
            return (len(non_zero_values) - len(filtered_values)) / len(non_zero_values) * 100 if len(non_zero_values) > 0 else 0
    
        # Función para calcular el percentil 80
        def calculate_75th_percentile(series):
            return series.quantile(0.8)
        
        """# Función para calcular el percentil 75 excluyendo los valores de 0 para tener un mejor tiempo estimado
        def calculate_75th_percentile_excluding_zeros(series):
            non_zero_values = series[series > 0]
            if len(non_zero_values) == 0:
                return 0
            return non_zero_values.quantile(0.75)"""
        
        # Creamos el nuevo conjunto de datos con las medias por coche y percentil 75
        grouped_data = self.data.groupby('tag_id').agg(
            mean_charge_time=('charge_duration', 'mean'),
            mean_energy=('energy', 'mean'),
            std_energy=('energy', 'std'),
            percentile_75_power_max=('power_max', calculate_75th_percentile),
            mean_time_between_charges=('time_between_charges', mean_excluding_zeros_and_outliers),
            std_time_between_charges=('time_between_charges', 'std'),
            wrong_date_probability=('time_between_charges', calculate_outlier_probability),
            mean_arrival_hour=('arrival_hour', 'mean'),
            std_arrival_hour=('arrival_hour', 'std'),
            num_charges = ('tag_id', 'count')
        ).reset_index().sort_values(by=['percentile_75_power_max'])

        # Redondear las columnas a 2 decimales y convertir las unidades a kWh y kW
        # Para las columnas de tiempo no dejamos decimales, solo para energy, power_max tampoco dejamos decimales
        grouped_data['mean_charge_time'] = grouped_data['mean_charge_time'].round()
        grouped_data['mean_energy'] = (grouped_data['mean_energy']/1000).round(2)
        grouped_data['percentile_75_power_max'] = (grouped_data['percentile_75_power_max']/1000).round()
        grouped_data['mean_time_between_charges'] = grouped_data['mean_time_between_charges'].fillna(0).round()
        grouped_data['std_time_between_charges'] = grouped_data['std_time_between_charges'].fillna(0).round(1)
        grouped_data['wrong_date_probability'] = grouped_data['wrong_date_probability'].round()
        grouped_data['mean_arrival_hour'] = grouped_data['mean_arrival_hour'].round()
        grouped_data['std_arrival_hour'] = grouped_data['std_arrival_hour'].fillna(0).round(1)

        # Agregar columna 'velocidad de carga' que ira en funcion del percentil 75
        def categorize_charge_speed(percentile_75_power_max):
            if percentile_75_power_max <= 50:
                return 'Slow'
            elif percentile_75_power_max <= 100:
                return 'Normal'
            else:
                return 'Fast'

        # Aplicar la categorización
        grouped_data['charge_speed'] = grouped_data['percentile_75_power_max'].apply(categorize_charge_speed)

        output_path = os.path.join(output_dir, 'processed_vehicle_data.csv')
        grouped_data.to_csv(output_path, index=False)
        print(f"CSV de datos procesados por vehículo generado: {output_path}")
        
        return grouped_data
    

    # Metodo que agrupa el dataset en funcion del percentil 75 de potencia y utilizarlo para futuras predicciones
    def generate_prediction_data(self, processed_data, output_dir):
        
        def mean_time_excluding_zeros(series):
            non_zero_values = series[series > 0]
            if len(non_zero_values) == 0:
                return 0
            return non_zero_values.mean()
        #Function to calculate weighted mean of a series based on weights on other series
        wm = lambda x: np.average(x, weights=processed_data.loc[x.index, 'num_charges'])
        # Creamos el nuevo conjunto de datos con las medias agrupando ya por el tipo de carga
        # Reminder: standard deviation is computed as the mean of the standard deviations of each vehicle
        # in the group. This is not exact, as there may be multiple observations per vehicle, and so different
        # weights may be applied to each observation. But it is quite close to the actual standard deviation.
        grouped_data = processed_data.groupby('charge_speed').agg(
            mean_charge_time=('mean_charge_time', wm),
            std_charge_time=('mean_charge_time', 'std'),
            mean_energy=('mean_energy', wm),
            std_energy=('mean_energy', 'std'),
            mean_time_between_charges=('mean_time_between_charges', mean_time_excluding_zeros),
            std_time_between_charges=('mean_time_between_charges', 'std'),
            wrong_date_probability=('wrong_date_probability', wm),
            mean_arrival_hour=('mean_arrival_hour', wm),
            std_arrival_hour=('mean_arrival_hour', 'std')
        ).reset_index().sort_values(by=['charge_speed'])

        # Anadir la columna con la cantidad de elementos en cada grupo
        grouped_data['count_in_group'] = processed_data.groupby('charge_speed')['charge_speed'].transform('size')

        # Redondear las columnas a 2 decimales y convertir las unidades a kWh y kW
        # Para las columnas de tiempo no dejamos decimales, solo para energy, power_max tampoco dejamos decimales
        grouped_data['mean_charge_time'] = grouped_data['mean_charge_time'].round()
        grouped_data['std_charge_time'] = grouped_data['std_charge_time'].round(3)
        grouped_data['mean_energy'] = (grouped_data['mean_energy']).round(2)
        grouped_data['std_energy'] = (grouped_data['std_energy']).round(3)
        grouped_data['mean_time_between_charges'] = grouped_data['mean_time_between_charges'].fillna(0).round()
        grouped_data['std_time_between_charges'] = grouped_data['std_time_between_charges'].fillna(0).round(1)
        grouped_data['wrong_date_probability'] = grouped_data['wrong_date_probability'].round()
        grouped_data['mean_arrival_hour'] = grouped_data['mean_arrival_hour'].round()
        grouped_data['std_arrival_hour'] = grouped_data['std_arrival_hour'].fillna(0).round(1)

        output_path = os.path.join(output_dir, 'prediction_data.csv')
        grouped_data.to_csv(output_path, index=False)
        print(f"CSV de datos procesados por percentil 75: {output_path}")


    # Metodo que agrupa en funcion de la potencia maxima en vez de por coche (no utilizado actualmente)
    def group_by_powerMax(self, data, output_dir):
        grouped_data = data.groupby('max_power_max').agg(
            vehicle_count = ('tag_id', 'size'),
            mean_charge_time=('mean_charge_time', 'mean'),
            mean_energy=('mean_energy', 'mean'),
            mean_time_between_charges=('mean_time_between_charges', 'mean')
        ).reset_index().sort_values(by=['max_power_max'])

        output_path = os.path.join(output_dir, 'prediction_data.csv')
        grouped_data.to_csv(output_path, index=False)
        print(f"CSV de datos para predicciones generado: {output_path}")

    # Metodo que busca la correlacion entre la potencia del vehiculo y la hora a la que llega a cargar (no utilizado actualmente)
    def analyze_gradients(self, output_dir):
        """Analizar los gradientes entre la hora de llegada y la potencia máxima"""
        if 'arrival_hour' not in self.data.columns:
            raise KeyError("La columna 'arrival_hour' no está presente en el DataFrame.")
        if 'power_max' not in self.data.columns:
            raise KeyError("La columna 'power_max' no está presente en el DataFrame.")
        
        # Preparar los datos para la regresión lineal
        X = self.data['arrival_hour'].values.reshape(-1, 1)
        y = self.data['power_max'].values.reshape(-1, 1)

        # Ajustar el modelo de regresión lineal
        model = LinearRegression()
        model.fit(X, y)

        # Predicciones
        y_pred = model.predict(X)

        # Visualización con gradientes
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='blue', label='Data Points')
        plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
        plt.xlabel('Arrival Hour')
        plt.ylabel('Max Power (W)')
        plt.title('Gradient Analysis between Arrival Hour and Max Power')
        plt.grid(True)
        plt.legend()

        # Guardar la imagen
        output_path = os.path.join(output_dir, 'gradient_analysis_arrival_hour_power_max.png')
        plt.savefig(output_path)
        plt.close()

        print(f"Imagen de análisis de gradientes guardada en: {output_path}")

        # Mostrar la pendiente (gradiente) de la regresión lineal
        gradient = model.coef_[0]
        intercept = model.intercept_
        print(f"Gradiente (Pendiente) de la regresión lineal: {gradient}")
        print(f"Intersección de la regresión lineal: {intercept}")

if __name__ == "__main__":
    dataProcessor = DataProcessor()
    dataProcessor.process_data()
