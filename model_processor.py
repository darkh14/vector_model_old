
from logger import ProcessorException as ProcessorException
import db_connector
import numpy as np

import pandas as pd
import os
import joblib
from job_processor import JobProcessor

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras

import matplotlib.pyplot as plt
import base64

DB_CONNECTOR = None


class ModelProcessor:

    def __init__(self, parameters):

        set_db_connector(parameters)


        # self._additional_data = {'periods': [],
        #                        'organisations': [],
        #                        'scenarios': [],
        #                        'years': [],
        #                        'x_indicators': [],
        #                        'y_indicators': []}

        # self._columns = {'x_columns': [],
        #                 'y_columns': []}

        self._parameters = parameters
        # self.X = np.array([[]])
        # self.X_scaled = np.array([[]])
        # self.y = np.array([])
        # self.X_pred = np.array([[]])
        # self.X_pd_pred = pd.DataFrame(np.array([[]]))
        # self.y_pd_pred = pd.DataFrame(np.array([[]]))
        # self.model = None
        # self._model_is_set = False
        # self.graph_file_name = 'graph.png'

    def load_data(self, raw_data, overwrite=False):

        data_processor = DataProcessor()

        data_processor.load_data(raw_data, overwrite=overwrite)




    # def prepare_data(self, for_prediction=False, inputs=None):
    #
    #     if not for_prediction:
    #         self.read_data()
    #         self.write_additional_data()
    #         inputs = self.inputs
    #     else:
    #         if not inputs:
    #             raise ProcessorException('Parameter "inputs" is not found')
    #         self.read_additional_data()
    #
    #     input_data_col = self._prepare_dataset_merge(inputs, self._additional_data['x_indicators'])
    #     output_data_col = None
    #     if not for_prediction:
    #         output_data_col = self._prepare_dataset_merge(self.outputs, self._additional_data['y_indicators'])
    #
    #     if not for_prediction:
    #         input_data_col, output_data_col = self._fill_empty_lines(input_data_col, output_data_col)
    #     else:
    #         input_data_col = input_data_col.fillna(0)
    #
    #     input_data_col = self._prepare_dataset_sort_one_hot(input_data_col)
    #     y = None
    #     if not for_prediction:
    #         output_data_col = self._prepare_dataset_sort_one_hot(output_data_col)
    #         self.db_connector.write_model_pd_data(input_data_col, output_data_col, rewrite=True)
    #         output_data_col.drop(['PortionNumber', 'Organisation', 'Scenario', 'year'], axis=1, inplace=True)
    #         col_changing = [col for col in self._additional_data['y_indicators'] if col.split(' ')[-1] == 'изменение']
    #         col_changing_months = np.array([[col + '_' + str(month) for month in range(1, 13)] for col in col_changing])
    #         col_changing_months = list(col_changing_months.flatten())
    #         output_data_col = output_data_col[col_changing_months]
    #         y = output_data_col.to_numpy()
    #         self.write_columns(input_data_col, output_data_col, rewrite=True)
    #
    #     # input_data_col.drop(['PortionNumber', 'Organisation', 'Scenario', 'year'], axis=1, inplace=True)
    #     input_data_col.drop(['PortionNumber', 'Scenario'], axis=1, inplace=True)
    #
    #     X = input_data_col.to_numpy()
    #
    #     if not for_prediction:
    #         self.db_connector.write_x_y(X, y, rewrite=True)
    #     else:
    #         self.X_pred = X
    #         self.X_pd_pred = input_data_col
    #

    #
    # def read_data(self, reset=False):
    #
    #     if reset or not self._data_is_set:
    #         if not self.db_connector:
    #             self.db_connector = db_connector.Connector(self._parameters, initialize=True)
    #
    #         self.inputs, self.outputs = self.db_connector.read_model_data()
    #         self._data_is_set = True
    #
    # def set_data(self, inputs, outputs):
    #     self.inputs = inputs
    #     self.outputs = outputs
    #     self._data_is_set = True
    #
    # def write_additional_data(self):
    #     if not self._data_is_set:
    #         raise ProcessorException('data is not set')
    #
    #     self.set_additional_data()
    #     self.db_connector.write_additional_model_data(self._additional_data, rewrite=True)
    #
    # def read_additional_data(self, reset=False):
    #
    #     if reset or not self._additional_data_is_set:
    #         if not self.db_connector:
    #             self.db_connector = db_connector.Connector(self._parameters, initialize=True)
    #         names_list = ['periods', 'organisations', 'scenarios', 'years', 'x_indicators', 'y_indicators']
    #         self._additional_data = self.db_connector.read_additional_data(names_list)
    #         self._additional_data_is_set = True
    #
    # def write_columns(self, inputs, outputs, rewrite=True):
    #     if not self._data_is_set:
    #         raise ProcessorException('data is not set')
    #
    #     self.set_columns(inputs, outputs)
    #     self.db_connector.write_additional_model_data(self._columns, rewrite=rewrite)
    #
    # def set_additional_data(self, reset=False):
    #
    #     if reset or not self._additional_data_is_set:
    #         p_inputs = pd.DataFrame(self.inputs)
    #         p_outputs = pd.DataFrame(self.outputs)
    #         x_periods = p_inputs['Period'].unique()
    #         y_periods = p_outputs['Period'].unique()
    #         x_organisations = p_inputs['Organisation'].unique()
    #         y_organisations = p_outputs['Organisation'].unique()
    #         x_scenarios = p_inputs['Scenario'].unique()
    #         y_scenarios = p_outputs['Scenario'].unique()
    #         x_indicators = p_inputs['Indicator'].unique()
    #         y_indicators = p_outputs['Indicator'].unique()
    #
    #         periods = self._make_x_y_list(x_periods, y_periods)
    #         organisations = self._make_x_y_list(x_organisations, y_organisations)
    #         scenarios = self._make_x_y_list(x_scenarios, y_scenarios)
    #
    #         years = map(lambda period: int(period.split('.')[2]), periods)
    #         years = list(set(years))
    #
    #         names_list = ['periods', 'organisations', 'scenarios', 'years', 'x_indicators', 'y_indicators']
    #
    #         values_list = [periods, organisations, scenarios, years, x_indicators, y_indicators]
    #
    #         self._additional_data = dict(zip(names_list, values_list))
    #
    #         self._additional_data_is_set = True
    #
    # def set_columns(self, inputs, outputs, reset=False):
    #
    #     if reset or not self._columns_is_set:
    #
    #         x_columns = list(inputs.columns)
    #         y_columns = list(outputs.columns)
    #
    #         names_list = ['x_columns', 'y_columns']
    #
    #         values_list = [x_columns, y_columns]
    #
    #         self._columns = dict(zip(names_list, values_list))
    #
    #         self._columns_is_set = True
    #
    # def read_columns(self, reset=False):
    #
    #     if reset or not self._columns_is_set:
    #         if not self.db_connector:
    #             self.db_connector = db_connector.Connector(self._parameters, initialize=True)
    #
    #         self._columns = self.db_connector.read_additional_data(['x_columns', 'y_columns'])
    #         self._columns_is_set = True
    #
    # def _prepare_dataset_merge(self, dataset, indicators):
    #
    #     pd_dataset = pd.DataFrame(dataset)
    #     pd_dataset = self._add_month_year_to_data(pd_dataset)
    #     dataset_col, raw_dataset_grouped = self._group_dataset(pd_dataset)
    #     raw_dataset_grouped = self._add_month_year_to_data(raw_dataset_grouped)
    #     raw_dataset_grouped.drop(['Period'], axis=1, inplace=True)
    #     dataset_col = self._merge_dataset(dataset_col, raw_dataset_grouped, indicators)
    #
    #     return dataset_col
    #
    # def _prepare_dataset_sort_one_hot(self, dataset):
    #     dataset = dataset.sort_values(by=['Organisation', 'Scenario', 'year'])
    #     fields_dict = {'Organisation': self._additional_data['organisations'],
    #                    'Scenario': self._additional_data['scenarios'],
    #                    'year': self._additional_data['years']}
    #     for field_name, field_values in fields_dict.items():
    #         dataset = self._one_hot_encode(dataset, field_name, field_values)
    #
    #     return dataset
    #
    # def _add_month_year_to_data(self, dataset):
    #     dataset['month'] = dataset['Period'].apply(self._get_month)
    #     dataset['year'] = dataset['Period'].apply(self._get_year)
    #
    #     return dataset
    #
    # @staticmethod
    # def _group_dataset(dataset):
    #
    #     data_col = dataset.groupby(['PortionNumber', 'Organisation', 'Scenario', 'year'], as_index=False).sum()
    #     data_col = data_col[['PortionNumber', 'Organisation', 'Scenario', 'year']]
    #
    #     raw_data_grouped = dataset.groupby(['PortionNumber',
    #                                         'Organisation',
    #                                         'Scenario',
    #                                         'Period',
    #                                         'Indicator'], as_index=False).sum()
    #     raw_data_grouped = raw_data_grouped[['PortionNumber',
    #                                          'Organisation',
    #                                          'Scenario',
    #                                          'Period',
    #                                          'Indicator',
    #                                          'Value']]
    #     return data_col, raw_data_grouped
    #
    # @staticmethod
    # def _merge_dataset(dataset, dataset_with_value, indicators):
    #
    #     months = np.arange(1, 13)
    #     for indicator in indicators:
    #         for month in months:
    #             dataset = pd.merge(dataset, dataset_with_value.loc[(dataset_with_value['Indicator'] == indicator)
    #                                                                & (dataset_with_value['month'] == month)],
    #                                on=['PortionNumber', 'Organisation', 'Scenario', 'year'], how='left')
    #             dataset.rename(columns={'Value': indicator + '_' + str(month)}, inplace=True)
    #             dataset.drop(['month'], axis=1, inplace=True)
    #             dataset.drop(['Indicator'], axis=1, inplace=True)
    #
    #     return dataset
    #
    # def _fill_empty_lines(self, input_dataset, output_dataset):
    #     for organisation in self._additional_data['organisations']:
    #         for scenario in self._additional_data['scenarios']:
    #             for year in self._additional_data['years']:
    #                 x_lines = input_dataset.loc[(input_dataset['Organisation'] == organisation)
    #                                              & (input_dataset['Scenario'] == scenario)
    #                                              & (input_dataset['year'] == year)]
    #
    #                 y_lines = output_dataset.loc[(output_dataset['Organisation'] == organisation)
    #                                               & (output_dataset['Scenario'] == scenario)
    #                                               & (output_dataset['year'] == year)]
    #
    #                 if x_lines.shape[0] == 1 and y_lines.shape[0] == 0:
    #                     row = {'Organisation': organisation, 'Scenario': scenario, 'year': year}
    #                     output_dataset = output_dataset.append(row, ignore_index=True)
    #                     # print('{} - {} - {}'.format(organisation, scenario, year))
    #
    #                 if x_lines.shape[0] == 0 and y_lines.shape[0] == 1:
    #                     row = {'Organisation': organisation, 'Scenario': scenario, 'year': year}
    #                     input_dataset = input_dataset.append(row, ignore_index=True)
    #                     # print('{} - {} - {}'.format(organisation, scenario, year))
    #
    #     input_dataset = input_dataset.fillna(0)
    #     output_dataset = output_dataset.fillna(0)
    #
    #     return input_dataset, output_dataset
    #
    # @staticmethod
    # def _one_hot_encode(dataset, field_name, indicators):
    #     for indicator in indicators:
    #         dataset[field_name.lower() + ' '
    #                 + str(indicator)] = dataset[field_name].apply(lambda x: 1 if x == indicator else 0)
    #     # dataset.drop([field_name], axis=1, inplace=True)
    #     return dataset
    #
    # @staticmethod
    # def _get_scaler(model_name='', renew=False, do_not_create=False):
    #
    #     scaler_filename = "scaler.save"
    #
    #     # if not os.path.exists("models"):
    #     #     os.mkdir('models')
    #     scaler = None
    #
    #     if not renew:
    #         if os.path.isfile(scaler_filename):
    #             scaler = joblib.load(scaler_filename)
    #         elif not do_not_create:
    #             scaler = MinMaxScaler()
    #     else:
    #         scaler = MinMaxScaler()
    #
    #     return scaler
    #
    # def _set_model(self, model_name='', renew=False, do_not_create=False):
    #
    #     folder_name = "model"
    #     if renew or not self._model_is_set:
    #         create = False
    #
    #         if not renew:
    #             if os.path.isdir(folder_name):
    #                 self.model = self.model = keras.models.load_model(folder_name)
    #             elif not do_not_create:
    #                 create = True
    #         else:
    #             create = True
    #
    #         if create:
    #             self.model = self._create_model(self.X.shape[1], self.y.shape[1])
    #
    # def _save_model(self):
    #
    #     folder_name = "model"
    #     self.model.save(folder_name)
    #
    # @staticmethod
    # def _create_model(inputs_number, outputs_number):
    #     model = Sequential()
    #     model.add(Dense(600, activation="relu", input_shape=(inputs_number,), name='dense_1'))
    #     model.add(Dense(400, activation="relu", name='dense_2'))
    #     model.add(Dense(outputs_number, activation="linear", name='dense_4'))
    #     return model
    #
    # @staticmethod
    # def _make_x_y_list(x_list, y_list):
    #     x_set = set(x_list)
    #     y_set = set(y_list)
    #     return list(x_set.union(y_set))
    #
    # @staticmethod
    # def _get_year(date_str):
    #     return int(date_str.split('.')[2])
    #
    # @staticmethod
    # def _get_month(date_str):
    #     return int(date_str.split('.')[1])
    #
    # def _make_graph(self, x, y, x_label, y_label):
    #
    #     fig, ax = plt.subplots()
    #
    #     ax.plot(x, y)# , label='y_test')
    #
    #     ax.set_xlabel(x_label)
    #     ax.set_ylabel(y_label)
    #     # ax.legend()
    #
    #     # plt.show()
    #     fig.savefig(self.graph_file_name)
    #
    # def read_graph_file(self):
    #
    #     f = open(self.graph_file_name, 'rb')
    #     result = f.read()
    #     f.close()
    #
    #     return result
    #
    # @staticmethod
    # def _get_dataframe_for_graph(x, x_indicator):
    #     columns = [x_indicator + '_' + str(number) for number in range(1, 13)]
    #     result = x[columns]
    #     result = result.sum(axis=1)
    #
    #     result = result/12
    #     return np.array(result)


class Model:

    def __init__(self, model_id, name='', organisation='', period=''):
        self.model_id = model_id

    def fit(self, parameters):

        epochs = parameters.get('epochs') or 10
        validation_split = parameters.get('validation_split') or 0.2

        if not self.db_connector:
            self.db_connector = db_connector.Connector(self._parameters, initialize=True)

        self.X, self.y = self.db_connector.read_x_y()
        scaler = self._get_scaler()
        self.X_scaled = scaler.fit_transform(self.X)

        self._set_model(renew=True)
        self.model.compile(optimizer='adam', loss='MeanSquaredError', metrics=['RootMeanSquaredError'])
        history = self.model.fit(self.X_scaled, self.y, epochs=epochs, verbose=2, validation_split=validation_split)

        self._save_model()

        return history.history

    def predict(self, inputs, indicators=None, get_graph=False, graph_data=None):
        self.inputs_pred = inputs
        self.prepare_data(for_prediction=True)
        self.read_columns()
        self._set_model()
        y = self.model.predict(self.X_pred)
        self.y_pd_pred = pd.DataFrame(y, columns=self._columns['y_columns'])

        graph_bin = None
        if get_graph:
            x_graph = self._get_dataframe_for_graph(self.X_pd_pred, graph_data['x_indicator'])
            y_graph = self._get_dataframe_for_graph(self.y_pd_pred, graph_data['y_indicator'])

            self._make_graph(x_graph, y_graph, graph_data['x_indicator'], graph_data['y_indicator'])

            graph_bin = self.read_graph_file()

        return self.y_pd_pred.to_dict('records'), graph_bin


class DataProcessor:

    def __init__(self):

        self._db_connector = DB_CONNECTOR

    def load_data(self, raw_data, overwrite=False):
        self._db_connector.write_raw_data(raw_data, overwrite=overwrite)


def load_data(parameters):

    raw_data = parameters.get('raw_data')

    if not raw_data:
        raise ProcessorException('raw data is not found in parameters')

    processor = ModelProcessor(parameters)
    processor.load_data(raw_data)

    return {'status': 'OK', 'error_text': '', 'description': 'model data loaded'}


@JobProcessor.job_processing
def fit(parameters):
    processor = ModelProcessor(parameters)
    history = processor.fit(parameters)
    return {'status': 'OK', 'error_text': '', 'description': 'model fitted', 'history': history}


def predict(parameters):
    inputs = parameters.get('inputs')
    if not inputs:
        raise ProcessorException('inputs not found in parameters')

    get_graph = parameters.get('get_graph')
    graph_data = parameters.get('graph_data')
    processor = ModelProcessor(parameters)
    prediction, graph_bin = processor.predict(inputs, get_graph=get_graph, graph_data=graph_data)

    result = dict(status='OK', error_text='', result=prediction, description='model data loaded')

    if graph_bin:
        result['graph_data'] = base64.b64encode(graph_bin).decode(encoding='utf-8')
    return result


def set_db_connector(parameters):
    global DB_CONNECTOR
    if not DB_CONNECTOR:
        DB_CONNECTOR = db_connector.Connector(parameters, initialize=True)
