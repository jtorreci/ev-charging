from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF
import os

class ReportGenerator:
    def __init__(self):
        pass

    def save_histograms(self, data, output_dir):
        """Crear histogramas para visualizar la distribución de los datos"""
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(data['energy'], bins=30, color='skyblue', edgecolor='black')
        ax1.set_title('Distribution of Energy')
        ax1.set_xlabel('Energy')
        ax1.set_ylabel('Frequency')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(data['battery_pct'], bins=30, color='lightgreen', edgecolor='black')
        ax2.set_title('Distribution of Battery Percentage')
        ax2.set_xlabel('Battery Percentage')
        ax2.set_ylabel('Frequency')

        ax3 = fig.add_subplot(gs[1, :])
        ax3.hist(data['power_max'], bins=30, color='lightcoral', edgecolor='black')
        ax3.set_title('Distribution of Power Max')
        ax3.set_xlabel('Power Max')
        ax3.set_ylabel('Frequency')

        plt.tight_layout()
        histograms_path = os.path.join(output_dir, 'histograms.png')
        plt.savefig(histograms_path)
        plt.close()
        return histograms_path

    def save_outliers_histogram(self, outliers_energy, outliers_power_max, outliers_battery_pct, output_dir):
        """Crear histogramas para visualizar la distribución de los outliers"""
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(outliers_energy, bins=30, color='skyblue', edgecolor='black')
        ax1.set_title('Outliers in Energy')
        ax1.set_xlabel('Energy')
        ax1.set_ylabel('Frequency')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(outliers_battery_pct, bins=30, color='lightgreen', edgecolor='black')
        ax2.set_title('Outliers in Battery Percentage')
        ax2.set_xlabel('Battery Percentage')
        ax2.set_ylabel('Frequency')

        ax3 = fig.add_subplot(gs[1, :])
        ax3.hist(outliers_power_max, bins=30, color='lightcoral', edgecolor='black')
        ax3.set_title('Outliers in Power Max')
        ax3.set_xlabel('Power Max')
        ax3.set_ylabel('Frequency')

        plt.tight_layout()
        outliers_histograms_path = os.path.join(output_dir, 'outliers_histograms.png')
        plt.savefig(outliers_histograms_path)
        plt.close()
        return outliers_histograms_path

    def save_date_errors_plot(self, invalid_dates, negative_durations):
        """Crear gráfico de errores en fechas"""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.bar(invalid_dates['original_index'], [1]*len(invalid_dates), color='red', label='Invalid Dates')
        ax.bar(negative_durations['original_index'], [1]*len(negative_durations), color='blue', label='Negative Durations')

        ax.set_xlabel('Original Index')
        ax.set_ylabel('Error')
        ax.set_title('Date Errors')
        ax.legend()

        plt.tight_layout()
        date_errors_path = os.path.join('output', 'date_errors.png')
        plt.savefig(date_errors_path)
        plt.close()
        return date_errors_path

    def plot_percentile_histogram(self, grouped_data, output_dir):
        """Generar un histograma para visualizar la distribución del percentil 75 de la potencia máxima."""
        plt.figure(figsize=(10, 6))
        plt.hist(grouped_data['percentile_75_power_max'], bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('Percentil 75 de Potencia Máxima (kW)')
        plt.ylabel('Frecuencia')
        plt.title('Distribución del Percentil 75 de Potencia Máxima')
        
        output_path = os.path.join(output_dir, 'percentile_75_power_max_histogram.png')
        plt.savefig(output_path)
        plt.close()
        
        print(f"Histograma de percentil 75 de potencia máxima generado: {output_path}")

    def time_between_charges_JATP(self, data, output_dir):
        plt.figure(figsize=(12, 8))

        # Filtrar valores menores o iguales a 0
        filtered_data = data[data['mean_time_between_charges'] > 0]

        # Agrupar por 'charge_speed' y generar la curva de densidad para cada grupo
        for charge_speed in filtered_data['charge_speed'].unique():
            subset = filtered_data[filtered_data['charge_speed'] == charge_speed]
            subset['mean_time_between_charges'].plot.kde(label=f'{charge_speed} charge speed', alpha=0.6)

        plt.xlabel('Mean Time Between Charges (hours)')
        plt.ylabel('Density')
        plt.title('Kernel Density Estimate of Time Between Charges by Charge Speed')
        plt.legend(title='Charge Speed')

        # Guardar el gráfico en el directorio de salida
        output_path = os.path.join(output_dir, 'time_between_charges_JATP.png')
        plt.savefig(output_path)
        plt.close()

        print(f"Gráfico de tiempo entre cargas generado: {output_path}")
        return

    
    def time_between_charges_histogram(self, data, output_dir):
        # Asegúrate de que 'charge_speed' esté definido en tu DataFrame
        if 'charge_speed' not in data.columns:
            raise ValueError("La columna 'charge_speed' no está definida en el DataFrame. Asegúrate de que 'group_by_tagID' se haya ejecutado correctamente.")

        # Crear una figura y ejes para el gráfico
        plt.figure(figsize=(12, 8))

        # Graficar la distribución de 'time_between_charges' para cada categoría de 'charge_speed'
        for speed in data['charge_speed'].unique():
            data = data[data['mean_time_between_charges'] > 0]
            subset = data[data['charge_speed'] == speed] 
            subset['mean_time_between_charges'].plot(kind='density', label=speed)

        # Configurar etiquetas y título del gráfico
        plt.xlabel('Time Between Charges (hours)')
        plt.ylabel('Density')
        plt.title('Density Plot of Time Between Charges by Charge Speed')
        plt.legend(title='Charge Speed')

        # Guardar el gráfico en el directorio de salida
        output_path = os.path.join(output_dir, 'time_between_charges_by_speed.png')
        plt.savefig(output_path)
        plt.close()

        print(f"Gráfico de tiempo entre cargas por velocidad de carga generado: {output_path}")