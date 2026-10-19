from DataProcessor import DataProcessor
from reportGenerator import ReportGenerator

import os

""" Before starting, we proceed to introduce the relevant data that will be used and their measurements:
        tag_id --> Car identifier (String)
        energy --> Total energy supplied in that charge (Wh)
        power --> Last recorded power of that charge (W), not useful
        power_max --> Maximum recorded power in that charge (W)
        battery_start --> Battery percentage at the beginning of the charge (%)
        battery_pct --> Battery percentage at the end of the charge (%)
    
    On the other hand, the following columns have been created for data processing:
        charge_duration --> Duration of that charge (minutes)
        mean_charge_time --> Average duration of the charge for that vehicle (minutes)
        mean_energy --> Average energy supplied to that vehicle (kWh)
        max_power_max --> Maximum power supplied to that vehicle (kW)
        mean_time_between_charges --> Average time between charges (hours), if the value is 0
                        it is because there is only one record of that car and therefore
                        the average time between charges cannot be calculated
"""
class Main:
    def __init__(self):
        self.file_path_pattern = os.path.join(os.path.dirname(__file__), '*.csv')
        self.output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(self.output_dir, exist_ok=True)  # Crear la carpeta output aquí
        self.data_processor = DataProcessor(self.file_path_pattern)
        self.report_generator = ReportGenerator()

    def run(self):
        # Cargar y concatenar los datos
        self.data_processor.load_and_concat_data()
        
        """Once the data is loaded, we will first clean it by removing records with a charge duration of less than 10 minutes. 
           This will also automatically remove all records with negative durations and/or invalid dates. 
           We will also remove records with a maximum supplied power of <= 0 as they are not useful for determining 
           which group the electric vehicle belongs to, and the power column as it is just the last supplied energy."""

        # Calcular la duración de cada transacción en minutos y contar los registros con duración menor a 10 minutos
        self.data_processor.data['charge_duration'] = (self.data_processor.data['stopped_at'] - self.data_processor.data['started_at']).dt.total_seconds() / 60
        #num_short_durations = len(self.data_processor.data[self.data_processor.data['charge_duration'] < 10])

        # Eliminar registros con duración menor a 10 minutos
        self.data_processor.data = self.data_processor.data[self.data_processor.data['charge_duration'] >= 10]

        # Eliminar registros con potencia máxima menor o igual a 0
        self.data_processor.data = self.data_processor.data[self.data_processor.data['power_max'] > 0]

        # Eliminar columna power ya que no es útil para nuestro análisis
        self.data_processor.data.drop(columns=['power'], inplace=True)

        # Guardar los histogramas como una imagen
        histograms_path = self.report_generator.save_histograms(self.data_processor.data, self.output_dir) 

        """Ahora que tenemos los datos eliminados pasaremos a ordenar por tag_id y rellenar los nulos de 
           las columnas battery_start y battery_pct"""
        
        # Generaremos el CSV ordenado y rellenando las columnas battery_start y battery_pct
        new_data_path = os.path.join(self.output_dir, 'clean_order_data.csv')
        clean_order_data = self.data_processor.order_fill_data()
        clean_order_data.to_csv(new_data_path, index=False)

        # Con todos los registros inválidos eliminados pasaremos a detectar posibles outliers en nuestros datos
        # Identificar outliers en las columnas especificadas, devulve solo la columna que se está trantando para ahorrar memoria.
        outliers_energy = self.data_processor.identify_outliers('energy')
        outliers_power_max = self.data_processor.identify_outliers('power_max')
        outliers_battery_pct = self.data_processor.identify_outliers('battery_pct')

        # Por el momento dejaremos los posibles outliers sin tratar ya que pueden ser datos útiles, generaremos un histograma para visualizarlos mejor
        outliers_histograms_path = self.report_generator.save_outliers_histogram(outliers_energy, outliers_power_max, outliers_battery_pct, self.output_dir)

        """ With the generated outliers, we will move on to the second part of this project, generating a CSV 
            grouped by vehicle. This will calculate the average time, average energy, and average time 
            between charges for each vehicle. Additionally, new columns will be assigned such as the most 
            common time the vehicle usually arrives to charge along with the deviation, and the type of 
            vehicle charge. To assign this value, we will use the 75th percentile which will replace power_max."""

        processed_data = self.data_processor.group_by_tagID(self.output_dir)
        
        self.report_generator.plot_percentile_histogram(processed_data, self.output_dir)
        self.report_generator.time_between_charges_JATP(processed_data, self.output_dir)
        #self.report_generator.time_between_charges_histogram(processed_data, self.output_dir);

        self.data_processor.generate_prediction_data(processed_data, self.output_dir)

        """ Finally, we will generate another 'predictions' CSV file that contains the number of vehicles
            grouped by maximum power, as well as the average energy supplied to them, average charge duration,
            and how often they might come on average."""
        
        #self.data_processor.group_by_powerMax(processed_data, self.output_dir)

        """ To perform a more complete and useful analysis for predictions, we will calculate if there is any type of correlation 
            between the groups of cars according to their charge and the arrival time at the charging station in order to 
            successfully predict the cars that will come to charge at a certain time.
        """

        #self.data_processor.analyze_gradients(self.output_dir)

# Ejecutar el main
if __name__ == "__main__":
    main = Main()
    main.run()
