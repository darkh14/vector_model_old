import settings_controller
from logger import ProcessorException as ProcessorException
import db_connector
import numpy as np

import pandas as pd
import os

from job_processor import JobProcessor

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras

import matplotlib.pyplot as plt
import base64

import pickle

DB_CONNECTOR = None


class ModelProcessor:

    def __init__(self, parameters):

        set_db_connector(parameters)
        self._data_processor = DataProcessor()

        self.model = None

        # self.X = np.array([[]])
        # self.X_scaled = np.array([[]])
        # self.y = np.array([])
        # self.X_pred = np.array([[]])
        # self.X_pd_pred = pd.DataFrame(np.array([[]]))
        # self.y_pd_pred = pd.DataFrame(np.array([[]]))
        # self.model = None
        # self._model_is_set = False
        # self.graph_file_name = 'graph.png'

    def load_data(self, parameters):

        raw_data = parameters.get('data')

        if not raw_data:
            raise ProcessorException('"data" is not found in parameters')

        overwrite = parameters.get('overwrite')

        self._data_processor.load_data(raw_data, overwrite=overwrite)

    def fit(self, parameters):
        model_description = parameters.get('model')

        if not model_description:
            raise ProcessorException('model is not in parameters')

        self.model = Model(model_description['id'],
                           model_description['name'],
                           model_description['x_indicators'],
                           model_description['y_indicators'])

        self.model.fit(parameters.get('epochs'), parameters.get('validation_split'))

    #

    #
    # def _set_model(self, model_name='', renew=False, do_not_create=False):
    #

    #
    # def _save_model(self):
    #

    #
    # @staticmethod
    # def _create_model(inputs_number, outputs_number):

    #
    # @staticmethod
    # def _make_x_y_list(x_list, y_list):
    #     x_set = set(x_list)
    #     y_set = set(y_list)
    #     return list(x_set.union(y_set))
    #

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

    def __init__(self, model_id, name='', x_indicators=[], y_indicators=[], need_to_update=False):
        self.model_id = model_id
        self._db_connector = DB_CONNECTOR
        self._data_processor = DataProcessor()
        description_from_db = self._data_processor.read_model_description_from_db(self.model_id)

        if description_from_db:
            self.name = description_from_db['name']
            self.x_indicators = description_from_db['x_indicators']
            self.y_indicators = description_from_db['y_indicators']
            self.periods = description_from_db['periods']
            self.organisations = description_from_db['organisations']
            self.scenarios = description_from_db['scenarios']
            self.x_columns = description_from_db['x_columns']
            self.y_columns = description_from_db['y_columns']
        else:
            self.name = name
            self.x_indicators = self._data_processor.get_indicator_ids(x_indicators)
            self.y_indicators = self._data_processor.get_indicator_ids(y_indicators)
            self.periods = []
            self.organisations = []
            self.scenarios = []
            self.x_columns = []
            self.y_columns = []

        self.need_to_update = not description_from_db or need_to_update

    def update_model(self, inputs, outputs):
        if self.need_to_update:
            data = inputs + outputs
            organisations, scenarios, periods = self._data_processor.get_additional_data(data)
            self.periods = periods
            self.organisations = organisations
            self.scenarios = scenarios

            model_description = {'model_id': self.model_id,
                                 'name': self.name,
                                 'x_indicators': self.x_indicators,
                                 'y_indicators': self.y_indicators,
                                 'periods': self.periods,
                                 'organisations': self.organisations,
                                 'scenarios': self.scenarios,
                                 'x_columns': self.x_columns,
                                 'y_columns': self.y_columns}

            self._data_processor.write_model_to_db(model_description)
            self.need_to_update = False

    def fit(self, epochs=100, validation_split=0.2, retrofit=False):

        inputs, outputs = self._data_processor.read_inputs_outputs_from_raw_data(self.x_indicators, self.y_indicators)
        self.update_model(inputs, outputs)

        additional_data = {'x_indicators': self.x_indicators,
                           'y_indicators': self.y_indicators,
                           'periods': self.periods,
                           'organisations': self.organisations,
                           'scenarios': self.scenarios,
                           'x_columns': self.x_columns,
                           'y_columns': self.y_columns}
        X, y, x_columns, y_columns = self._data_processor.get_X_y_for_fitting(inputs, outputs, additional_data)
        self.x_columns = x_columns
        self.y_columns = y_columns

        scaler = self._get_scaler(retrofit=retrofit)
        X = scaler.fit_transform(X)

        # self._set_model(renew=True)
        # self.model.compile(optimizer='adam', loss='MeanSquaredError', metrics=['RootMeanSquaredError'])
        # history = self.model.fit(self.X_scaled, self.y, epochs=epochs, verbose=2, validation_split=validation_split)
        #
        # self._save_model()
        #
        self._data_processor.write_scaler(self.model_id, scaler)
        # return history.history
        return []

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

    def _get_scaler(self, retrofit=False):

        if retrofit:
            scaler = self._data_processor.read_scaler(self.model_id)
            if not scaler:
                scaler = MinMaxScaler()
        else:
            scaler = MinMaxScaler()

        return scaler

    def _get_indicator_id(self, indicator, report_type):

        indicator_from_db = self._db_connector.read_indicator_from_name_type(indicator, report_type)

        if not indicator_from_db:
            indicator_id = 'ind_' + settings_controller.get_id()
            self._db_connector.write_indicator(indicator_id, indicator, report_type)
        else:
            indicator_id = indicator_from_db['indicator_id']

        return indicator_id

    def get_inner_model(self, retrofit=False):
        pass
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

        #     folder_name = "model"
        #     self.model.save(folder_name)


class DataProcessor:

    def __init__(self):

        self._db_connector = DB_CONNECTOR

    def load_data(self, raw_data, overwrite=False):
        raw_data = self.add_indicators_to_raw_data(raw_data)
        self._db_connector.write_raw_data(raw_data, overwrite=overwrite)

    def add_indicators_to_raw_data(self, raw_data):
        for line in raw_data:
            indicator_id = self._get_indicator_id(line['indicator'], line['report_type'])
            line['indicator_id'] = indicator_id

        return raw_data

    def read_model_description_from_db(self, model_id):
        return self._db_connector.read_model_description(model_id)

    def get_indicator_ids(self, indicators):
        result = [self._db_connector.read_indicator_from_name_type(ind_line['indicator'],
                                                                   ind_line['report_type'])['indicator_id']
                  for ind_line in indicators]
        return result

    def get_indicator_from_name_type(self, indicator, report_type):
        indicator_line = self._db_connector.read_indicator_from_name_type(indicator, report_type)
        if not indicator_line:
            indicator_id = 'ind_' + settings_controller.get_id()
            indicator_line = {'indicator_id': indicator_id, 'indicator': indicator, 'report_type': report_type}
            self._db_connector.write_indicator(indicator_line)

        return indicator_line

    def read_inputs_outputs_from_raw_data(self, x_indicators, y_indicators):
        inputs = self._db_connector.read_data_with_indicators_filter(x_indicators)
        outputs = self._db_connector.read_data_with_indicators_filter(y_indicators)
        return inputs, outputs

    @staticmethod
    def get_additional_data(raw_data):
        pd_data = pd.DataFrame(raw_data)
        organisations, scenarios, periods = list(pd_data['organisation'].unique()),\
                                            list(pd_data['scenario'].unique()), \
                                            list(pd_data['period'].unique())
        return organisations, scenarios, periods

    def write_model_to_db(self, model_description):
        self._db_connector.write_model_description(model_description)

    def get_X_y_for_fitting(self, inputs, outputs, additional_data):

        input_data_col = self._prepare_dataset_merge(inputs, additional_data['x_indicators'])
        output_data_col = self._prepare_dataset_merge(outputs, additional_data['y_indicators'])

        additional_data['years'] = list(set([self._get_year(period) for period in additional_data['periods']]))
        input_data_col, output_data_col = self._fill_empty_lines(input_data_col, output_data_col, additional_data)

        input_data_col = self._prepare_dataset_sort_one_hot(input_data_col, additional_data)

        input_data_col.drop(['organisation', 'scenario', 'year'], axis=1, inplace=True)
        output_data_col.drop(['organisation', 'scenario', 'year'], axis=1, inplace=True)
        x_columns = list(input_data_col.columns)
        y_columns = list(output_data_col.columns)

        X = input_data_col.to_numpy()
        y = output_data_col.to_numpy()

        return X, y, x_columns, y_columns

    def write_columns(self, model_id, x_columns, y_columns):
        self._db_connector.write_model_columns(model_id, x_columns, y_columns)

    def write_scaler(self, model_id, scaler):
        scaler_packed = pickle.dumps(scaler, protocol=pickle.HIGHEST_PROTOCOL)
        self._db_connector.write_model_scaler(model_id, scaler_packed)

    def read_scaler(self, model_id):
        model_description = self.read_model_description_from_db(model_id)
        scaler_packed = model_description['scaler']
        return pickle.loads(scaler_packed)

    def write_inner_model(self, model_id, inner_model):
        scaler_packed = pickle.dumps(inner_model, protocol=pickle.HIGHEST_PROTOCOL)
        self._db_connector.write_model_scaler(model_id, scaler_packed)

    def read_inner_model(self, model_id):
        model_description = self.read_model_description_from_db(self.model_id)
        scaler_packed = model_description['scaler']
        return pickle.loads(scaler_packed)

    def _prepare_dataset_merge(self, dataset, indicators):

        pd_dataset = pd.DataFrame(dataset)
        pd_dataset = self._add_month_year_to_data(pd_dataset)

        pd_dataset.drop(['_id', 'indicator', 'report_type'], axis=1, inplace=True)
        pd_dataset.rename({'indicator_id': 'indicator'}, axis=1, inplace=True)

        dataset_col, raw_dataset_grouped = self._group_dataset(pd_dataset)
        raw_dataset_grouped = self._add_month_year_to_data(raw_dataset_grouped)
        raw_dataset_grouped.drop(['period'], axis=1, inplace=True)
        dataset_col = self._merge_dataset(dataset_col, raw_dataset_grouped, indicators)

        return dataset_col

    def _prepare_dataset_sort_one_hot(self, dataset, additional_data):
        dataset = dataset.sort_values(by=['organisation', 'scenario', 'year'])
        fields_dict = {'organisation': additional_data['organisations'],
                       'year': additional_data['years']}
        for field_name, field_values in fields_dict.items():
            dataset = self._one_hot_encode(dataset, field_name, field_values)

        return dataset

    def _add_month_year_to_data(self, dataset):
        dataset['month'] = dataset['period'].apply(self._get_month)
        dataset['year'] = dataset['period'].apply(self._get_year)

        return dataset

    @staticmethod
    def _merge_dataset(dataset, dataset_with_value, indicators):

        months = np.arange(1, 13)
        for indicator in indicators:
            for month in months:
                dataset = pd.merge(dataset, dataset_with_value.loc[(dataset_with_value['indicator'] == indicator)
                                                                   & (dataset_with_value['month'] == month)],
                                   on=['organisation', 'scenario', 'year'], how='left')
                dataset.rename(columns={'value': indicator + '_' + str(month)}, inplace=True)
                dataset.drop(['month'], axis=1, inplace=True)
                dataset.drop(['indicator'], axis=1, inplace=True)

        return dataset

    @staticmethod
    def _group_dataset(dataset):

        data_col = dataset.groupby(['organisation', 'scenario', 'year'], as_index=False).sum()
        data_col = data_col[['organisation', 'scenario', 'year']]

        raw_data_grouped = dataset.groupby(['organisation',
                                            'scenario',
                                            'period',
                                            'indicator'], as_index=False)
        raw_data_grouped = raw_data_grouped.sum()
        raw_data_grouped_count = raw_data_grouped.count()

        raw_data_grouped['value'] = raw_data_grouped['value']/raw_data_grouped_count['value']

        raw_data_grouped = raw_data_grouped[['organisation',
                                             'scenario',
                                             'period',
                                             'indicator',
                                             'value']]
        return data_col, raw_data_grouped

    @staticmethod
    def _get_year(date_str):
        return int(date_str.split('.')[2])

    @staticmethod
    def _get_month(date_str):
        return int(date_str.split('.')[1])

    @staticmethod
    def _fill_empty_lines(input_dataset, output_dataset, additional_data):
        for organisation in additional_data['organisations']:
            for scenario in additional_data['scenarios']:
                for year in additional_data['years']:
                    x_lines = input_dataset.loc[(input_dataset['organisation'] == organisation)
                                                 & (input_dataset['scenario'] == scenario)
                                                 & (input_dataset['year'] == year)]

                    y_lines = output_dataset.loc[(output_dataset['organisation'] == organisation)
                                                  & (output_dataset['scenario'] == scenario)
                                                  & (output_dataset['year'] == year)]

                    if x_lines.shape[0] == 1 and y_lines.shape[0] == 0:
                        row = {'organisation': organisation, 'scenario': scenario, 'year': year}
                        output_dataset = output_dataset.append(row, ignore_index=True)
                        # print('{} - {} - {}'.format(organisation, scenario, year))

                    if x_lines.shape[0] == 0 and y_lines.shape[0] == 1:
                        row = {'organisation': organisation, 'scenario': scenario, 'year': year}
                        input_dataset = input_dataset.append(row, ignore_index=True)
                        # print('{} - {} - {}'.format(organisation, scenario, year))

        input_dataset = input_dataset.fillna(0)
        output_dataset = output_dataset.fillna(0)

        return input_dataset, output_dataset

    @staticmethod
    def _one_hot_encode(dataset, field_name, indicators):
        for indicator in indicators:
            dataset[field_name.lower() + ' '
                    + str(indicator)] = dataset[field_name].apply(lambda x: 1 if x == indicator else 0)
        # dataset.drop([field_name], axis=1, inplace=True)
        return dataset


def load_data(parameters):

    processor = ModelProcessor(parameters)
    processor.load_data(parameters)

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
