import settings_controller
from logger import ProcessorException as ProcessorException
import db_connector
import numpy as np

import pandas as pd
import os
import math
from abc import ABCMeta, abstractmethod

from job_processor import JobProcessor

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from keras.wrappers.scikit_learn import KerasRegressor
import eli5
from eli5.sklearn import PermutationImportance
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

from tensorflow import keras

import matplotlib.pyplot as plt
import matplotlib
import base64

import pickle
import zipfile
import shutil
import datetime

DB_CONNECTOR = None


class ModelProcessor:

    def __init__(self, parameters):

        set_db_connector(parameters)
        self._data_processor = DataProcessor()

        self.model = None

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

        need_to_update = bool(model_description.get('need_to_update'))

        self.model = self._get_model(model_description)

        retrofit = parameters.get('retrofit') and not need_to_update
        date_from = parameters.get('date_from')

        history = self.model.fit(epochs=parameters.get('epochs'),
                                 validation_split=parameters.get('validation_split'),
                                 retrofit=retrofit,
                                 date_from=date_from)

        return history

    def predict(self, parameters):

        model_description = parameters.get('model')

        if not model_description:
            raise ProcessorException('model is not in parameters')

        self.model = self._get_model(model_description)

        inputs = parameters.get('inputs')
        if not inputs:
            raise ProcessorException('inputs not found in parameters')

        get_graph = parameters.get('get_graph')
        graph_data = parameters.get('graph_data')

        prediction, indicator_description, graph_bin = self.model.predict(inputs,
                                                                          get_graph=get_graph,
                                                                          graph_data=graph_data)

        return prediction, indicator_description, graph_bin

    def calculate_feature_importances(self, parameters):

        model_description = parameters.get('model')

        if not model_description:
            raise ProcessorException('model is not in parameters')

        self.model = self._get_model(model_description)

        result = self.model.calculate_feature_importances(date_from=parameters.get('date_from'),
                                                          epochs=parameters.get('epochs'))

        return result

    def get_feature_importances(self, parameters):

        model_description = parameters.get('model')

        if not model_description:
            raise ProcessorException('model is not in parameters')

        self.model = self._get_model(model_description)

        result, graph_bin = self.model.get_feature_importances(parameters.get('get_graph'))

        return result, graph_bin

    def get_rsme(self, parameters):

        model_description = parameters.get('model')

        if not model_description:
            raise ProcessorException('model is not in parameters')

        self.model = self._get_model(model_description)

        rsme, mspe = self.model.get_rsme()

        return rsme, mspe

    @staticmethod
    def _get_model(model_description):

        need_to_update = bool(model_description.get('need_to_update'))

        model_type = model_description.get('type')
        if not model_type:
            raise ProcessorException('model type not in model description')

        model_class = None
        if model_type == 'neural_network':
            model_class = NeuralNetworkModel
        elif model_type == 'linear_regression':
            model_class = LinearModel

        if not model_class:
            raise ProcessorException('model type "{}" is not supported'.format(model_type))

        model = model_class(model_description['id'],
                            model_description.get('name'),
                            model_description.get('x_indicators'),
                            model_description.get('y_indicators'),
                            need_to_update=need_to_update,
                            model_filter=model_description.get('filter'))

        return model


class BaseModel:

    __metaclass__ = ABCMeta
    type = ''

    def __init__(self, model_id, name='', x_indicators=None, y_indicators=None, need_to_update=False,
                 model_filter=None):
        self.model_id = model_id
        self._db_connector = DB_CONNECTOR
        self._data_processor = DataProcessor()

        description_from_db = self._data_processor.read_model_description_from_db(self.model_id)

        if description_from_db:
            self.name = description_from_db['name']
            self.filter = description_from_db.get('filter')
            self.x_indicators = description_from_db['x_indicators']
            self.y_indicators = description_from_db['y_indicators']
            self.periods = description_from_db['periods']
            self.organisations = description_from_db['organisations']
            self.scenarios = description_from_db['scenarios']
            self.x_columns = description_from_db['x_columns']
            self.y_columns = description_from_db['y_columns']
            self.feature_importances = description_from_db.get('feature_importances')
        else:
            self.name = name
            self.filter = None
            self.x_indicators = []
            self.y_indicators = []
            self.periods = []
            self.organisations = []
            self.scenarios = []
            self.x_columns = []
            self.y_columns = []
            self.feature_importances = []

        if x_indicators:
            self.x_indicators = self._data_processor.get_indicators_data_from_parameters(x_indicators)

        if y_indicators:
            self.y_indicators = self._data_processor.get_indicators_data_from_parameters(y_indicators)

        if model_filter:
            self.filter = model_filter

        self._inner_model = None
        self._retrofit = False

        self.need_to_update = not description_from_db or need_to_update
        self.graph_file_name = 'graph.png'
        self.graph_fi_file_name = 'fi_graph.png'

    def update_model(self, data):
        if self.need_to_update:
            organisations, scenarios, periods = self._data_processor.get_additional_data(data)
            self.periods = periods
            self.organisations = organisations
            self.scenarios = scenarios

            model_description = {'model_id': self.model_id,
                                 'name': self.name,
                                 'filter': self.filter,
                                 'x_indicators': self.x_indicators,
                                 'y_indicators': self.y_indicators,
                                 'periods': self.periods,
                                 'organisations': self.organisations,
                                 'scenarios': self.scenarios,
                                 'x_columns': self.x_columns,
                                 'y_columns': self.y_columns,
                                 'feature_importances': self.feature_importances}

            self._data_processor.write_model_to_db(model_description)
            self.need_to_update = False

    @abstractmethod
    def fit(self, retrofit=False, date_from=None):
        """method for fitting model"""

    @abstractmethod
    def predict(self, inputs, get_graph=False, graph_data=None):
        """method for predicting data from model"""

    @abstractmethod
    def calculate_feature_importances(self, date_from=None, epochs=1000, retrofit=False, validation_split=0.2):
        """method for calculating feature importances"""

    def _calculate_fi_from_model(self, fi_model, x, y, x_columns):
        perm = PermutationImportance(fi_model, random_state=42).fit(x, y)

        fi = pd.DataFrame(perm.feature_importances_, columns=['feature_importance'])
        fi['feature'] = x_columns
        fi = fi.sort_values(by='feature_importance', ascending=False)
        fi['indicator'] = fi['feature'].apply(self._data_processor.get_indicator_name)
        fi['report_type'] = fi['feature'].apply(self._data_processor.get_indicator_report_type)

        fi = fi.to_dict('records')
        self._data_processor.write_feature_importances(self.model_id, fi)

        return fi

    def get_feature_importances(self, get_graph=False):
        fi = self._data_processor.read_feature_importances(self.model_id)
        if not fi:
            raise ProcessorException('Feature importances is not calculated')
        graph_bin = None
        if get_graph:
            graph_bin = self._get_fi_graph_bin(fi)

        return fi, graph_bin

    def get_rsme(self):
        rsme = self._data_processor.read_model_field(self.model_id, 'rsme')
        mspe = self._data_processor.read_model_field(self.model_id, 'mspe')

        if not rsme and rsme != 0:
            raise ProcessorException('RSME is not calculated')

        if not mspe and mspe != 0:
            raise ProcessorException('MSPE is not calculated')

        return rsme, mspe

    def _get_scaler(self, retrofit=False, is_out=False):

        if retrofit:
            scaler = self._data_processor.read_scaler(self.model_id, is_out)
            if not scaler:
                scaler = StandardScaler()
        else:
            scaler = StandardScaler()

        return scaler

    def _get_graph_bin(self, data, graph_data):

        x_graph, y_graph = self._get_dataframe_for_graph(data, graph_data['x_indicator_id'],
                                                         graph_data['y_indicator_id'])

        x_indicator_descr = self._data_processor.get_indicator_description_from_id(graph_data['x_indicator_id'])
        x_label = x_indicator_descr['report_type'] + '\n' + x_indicator_descr['name']
        y_indicator_descr = self._data_processor.get_indicator_description_from_id(graph_data['y_indicator_id'])
        y_label = y_indicator_descr['report_type'] + '\n' + y_indicator_descr['name']
        self._make_graph(x_graph, y_graph, x_label, y_label)

        graph_bin = self._read_graph_file()

        return graph_bin

    def _make_graph(self, x, y, x_label, y_label):

        x_max = max(x.max(), -(x.min()))
        x_mul = math.floor(math.log10(x_max))
        x_mul = math.floor(x_mul/3)*3
        x_mul = max(x_mul, 0)

        x = x*10**(-x_mul)

        y_max = max(y.max(), -(y.min()))
        y_mul = math.floor(math.log10(y_max))
        y_mul = math.floor(y_mul/3)*3
        y_mul = max(y_mul, 0)

        y = y*10**(-y_mul)

        fig, ax = plt.subplots()

        ax.plot(x, y) # , label='y_test')

        ax.set_xlabel(x_label + '\n' + '\\ {}'.format(10**x_mul))
        ax.set_ylabel(y_label + '\n' + '\\ {}'.format(10**y_mul))
        # ax.legend()

        fig.set_figwidth(8)  # ширина и
        fig.set_figheight(8)  # высота "Figure"

        ax.grid()

        # plt.show()
        fig.savefig(self.graph_file_name)

    def _read_graph_file(self, fi_graph=False):

        f = open(self.graph_fi_file_name if fi_graph else self.graph_file_name, 'rb')
        result = f.read()
        f.close()

        return result

    def _get_dataframe_for_graph(self, data, x_indicator_id, y_indicator_id):

        x_indicator_descr = self._data_processor.get_indicator_description_from_id(x_indicator_id)

        x_columns = []
        for col in self.x_columns:
            col_list = col.split('_')
            if len(col_list)==6:
                continue

            if col_list[1] == x_indicator_descr['short_id']:
                x_columns.append(col)

        y_indicator_descr = self._data_processor.get_indicator_description_from_id(y_indicator_id)
        y_columns = []
        for col in self.y_columns:
            col_list = col.split('_')
            if len(col_list)==6:
                continue

            if col_list[1] == y_indicator_descr['short_id']:
                y_columns.append(col)

        data = data[x_columns + y_columns].copy()

        data['x'] = data[x_columns].apply(sum, axis=1)

        data['y'] = data[y_columns].apply(sum, axis=1)

        data = data.drop(x_columns + y_columns, axis=1)
        data = data.sort_values(by=['x'])

        data['count'] = 1

        data = data.groupby(by=['x'], as_index=False).sum()
        data['y'] = data['y']/data['count']

        return np.array(data['x']), np.array(data['y'])

    def _get_fi_graph_bin(self, fi):

        values = [line['feature_importance']for line in fi]
        indexes = list(range(1, len(values) + 1))
        self._make_fi_graph(values, indexes)

        graph_bin = self._read_graph_file(fi_graph=True)

        return graph_bin

    def _make_fi_graph(self, values, indexes):

        fig, ax = plt.subplots()

        ax.bar(indexes, values) # , label='y_test')

        fig.set_figwidth(8)  # ширина и
        fig.set_figheight(8)  # высота "Figure"

        ax.grid()
        ax.set_xlim(xmin=indexes[0]-0.5, xmax=indexes[-1]+0.5)

        locator = matplotlib.ticker.MultipleLocator(base=1)

        ax.xaxis.set_major_locator(locator)

        # plt.show()
        fig.savefig(self.graph_fi_file_name)

    @staticmethod
    def _calculate_mspe(y_true, y_pred):

        eps = np.zeros(y_true.shape)
        eps[:] = 0.0001
        y_p = np.c_[abs(y_true), abs(y_pred), eps]
        y_p = np.max(y_p, axis=1).reshape(-1, 1)

        return np.sqrt(np.nanmean(np.square(((y_true - y_pred) / y_p))))


class NeuralNetworkModel(BaseModel):

    type = 'neural_network'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._epochs = 0
        self._validation_split = 0.2

        self._temp_input = None

    def fit(self, epochs=100, validation_split=0.2, retrofit=False, date_from=None):

        if not retrofit:
            date_from = None
        else:
            date_from = datetime.datetime.strptime(date_from, '%d.%m.%Y')

        indicator_filter = [ind_data['id'] for ind_data in self.x_indicators + self.y_indicators]

        data = self._data_processor.read_raw_data(indicator_filter, date_from=date_from)
        if retrofit and self.need_to_update:
            raise ProcessorException('Model can not be updated when retrofit')

        self.update_model(data)

        additional_data = {'x_indicators': self.x_indicators,
                           'y_indicators': self.y_indicators,
                           'periods': self.periods,
                           'organisations': self.organisations,
                           'scenarios': self.scenarios,
                           'x_columns': self.x_columns,
                           'y_columns': self.y_columns}
        # encode_fields = {'organisation': 'organisations', 'year': 'years', 'month': 'months'}
        encode_fields = None
        x, y, x_columns, y_columns = self._data_processor.get_x_y_for_fitting(data, additional_data, encode_fields,
                                                                              self.filter)

        self.x_columns = x_columns
        self.y_columns = y_columns
        if not retrofit:
            self._data_processor.write_columns(self.model_id, x_columns, y_columns)

        # x_scaler = self._get_scaler(retrofit=retrofit)
        # x_sc = x_scaler.fit_transform(x)
        #
        # y_scaler = self._get_scaler(retrofit=retrofit, is_out=True)
        # y_sc = y_scaler.fit_transform(y)

        inner_model = self._get_inner_model(x.shape[1], y.shape[1], retrofit=retrofit)

        self._inner_model = inner_model
        self._epochs = epochs or 1000
        self._validation_split = validation_split or 0.2

        # normalizer = inner_model.layers[0]
        # normalizer.adapt(x)

        inner_model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='MeanSquaredError',
                            metrics=['RootMeanSquaredError'])
        history = inner_model.fit(x, y, epochs=self._epochs, verbose=2, validation_split=self._validation_split)

        self._inner_model = inner_model

        # self._data_processor.write_scaler(self.model_id, x_scaler)
        # self._data_processor.write_scaler(self.model_id, y_scaler, is_out=True)

        y_pred = inner_model.predict(x)

        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mspe = self._calculate_mspe(y, y_pred)
        print("RMSE: {}".format(rmse))
        print("MSPE: {}".format(mspe))

        self._data_processor.write_model_field(self.model_id, 'rsme', rmse)
        self._data_processor.write_model_field(self.model_id, 'mspe', mspe)

        self._data_processor.write_inner_model(self.model_id, self._inner_model)

        return history.history

    def predict(self, inputs, get_graph=False, graph_data=None):

        data = pd.DataFrame(inputs)
        additional_data = {'x_indicators': self.x_indicators,
                           'y_indicators': self.y_indicators,
                           'periods': self.periods,
                           'organisations': self.organisations,
                           'scenarios': self.scenarios,
                           'x_columns': self.x_columns,
                           'y_columns': self.y_columns}

        # encode_fields = {'organisation': 'organisations', 'year': 'years', 'month': 'months'}
        encode_fields = None
        x, x_pd = self._data_processor.get_x_for_prediction(data, additional_data, encode_fields)

        inner_model = self._get_inner_model(retrofit=True)

        y = inner_model.predict(x)

        data = x_pd.copy()
        data[self.y_columns] = y

        graph_bin = None

        if get_graph:
            graph_bin = self._get_graph_bin(data, graph_data)

        outputs = data.drop(self.x_columns, axis=1)

        indicators_description = self.x_indicators + self.y_indicators

        return outputs.to_dict('records'), indicators_description, graph_bin

    def calculate_feature_importances(self, date_from=None, epochs=1000, retrofit=False, validation_split=0.2):

        data = self._data_processor.read_raw_data(self.x_indicators + self.y_indicators, date_from)
        additional_data = {'x_indicators': self.x_indicators,
                           'y_indicators': self.y_indicators,
                           'periods': self.periods,
                           'organisations': self.organisations,
                           'scenarios': self.scenarios,
                           'x_columns': self.x_columns,
                           'y_columns': self.y_columns}
        x, y, x_columns, y_columns = self._data_processor.get_x_y_for_fitting(data, additional_data)

        self._temp_input = x
        self._inner_model = self._get_inner_model(len(self.x_columns), len(self.y_columns), retrofit=retrofit)

        epochs = epochs or 1000
        validation_split = validation_split or 0.2

        fi_model = KerasRegressor(build_fn=self._get_model_for_feature_importances,
                                  epochs=epochs,
                                  verbose=2,
                                  validation_split=validation_split)
        fi_model.fit(x, y)

        fi = self._calculate_fi_from_model(fi_model, x, y, x_columns)

        return fi

    def _calculate_fi_from_model(self, fi_model, x, y, x_columns):
        perm = PermutationImportance(fi_model, random_state=42).fit(x, y)

        fi = pd.DataFrame(perm.feature_importances_, columns=['feature_importance'])
        fi['feature'] = x_columns
        fi = fi.sort_values(by='feature_importance', ascending=False)
        fi['indicator'] = fi['feature'].apply(self._data_processor.get_indicator_name)
        fi['report_type'] = fi['feature'].apply(self._data_processor.get_indicator_report_type)

        fi = fi.to_dict('records')
        self._data_processor.write_feature_importances(self.model_id, fi)

        return fi

    def _get_scaler(self, retrofit=False, is_out=False):

        if retrofit:
            scaler = self._data_processor.read_scaler(self.model_id, is_out)
            if not scaler:
                scaler = StandardScaler()
        else:
            scaler = StandardScaler()

        return scaler

    def _get_inner_model(self, inputs_number=0, outputs_number=0, retrofit=False):

        if retrofit:
            inner_model = self._data_processor.read_inner_model(self.model_id)
            if not inner_model:
                inner_model = self._create_inner_model(inputs_number, outputs_number)
        else:
            inner_model = self._create_inner_model(inputs_number, outputs_number)

        self._inner_model = inner_model

        return inner_model

    def _get_model_for_feature_importances(self):
        model_copy = clone_model(self._inner_model) # self._create_inner_model(len(self.x_columns), len(self.y_columns))
        model_copy.layers[0].adapt(self._temp_input)
        model_copy.compile(optimizer=optimizers.Adam(learning_rate=0.01), loss='MeanSquaredError',
                           metrics=['RootMeanSquaredError'])
        return model_copy

    @staticmethod
    def _create_inner_model(inputs_number, outputs_number):

        model = Sequential()
        # normalizer = Normalization(axis=-1)
        # model.add(normalizer)
        model.add(Dense(300, activation="relu", input_shape=(inputs_number,), name='dense_1'))
        model.add(Dense(250, activation="relu", name='dense_2'))
        model.add(Dense(100, activation="relu",  name='dense_3'))
        model.add(Dense(30, activation="relu", name='dense_4'))
        model.add(Dense(outputs_number, activation="linear", name='dense_last'))

        return model


class LinearModel(BaseModel):

    type = 'linear_regression'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, **kwargs):

        indicator_filter = [ind_data['id'] for ind_data in self.x_indicators + self.y_indicators]

        data = self._data_processor.read_raw_data(indicator_filter)

        self.update_model(data)

        additional_data = {'x_indicators': self.x_indicators,
                           'y_indicators': self.y_indicators,
                           'periods': self.periods,
                           'organisations': self.organisations,
                           'scenarios': self.scenarios,
                           'x_columns': self.x_columns,
                           'y_columns': self.y_columns}
        x, y, x_columns, y_columns = self._data_processor.get_x_y_for_fitting(data, additional_data)

        self.x_columns = x_columns
        self.y_columns = y_columns

        self._data_processor.write_columns(self.model_id, x_columns, y_columns)

        x_scaler = self._get_scaler(retrofit=False)
        x_sc = x_scaler.fit_transform(x)

        y_scaler = self._get_scaler(retrofit=False, is_out=True)
        y_sc = y_scaler.fit_transform(y)

        inner_model = self._get_inner_model()

        self._inner_model = inner_model

        inner_model.fit(x_sc, y_sc)

        self._data_processor.write_scaler(self.model_id, x_scaler)
        self._data_processor.write_scaler(self.model_id, y_scaler, is_out=True)

        y_pred_sc = inner_model.predict(x_sc)

        y_pred = y_scaler.inverse_transform(y_pred_sc)

        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mape = mean_absolute_percentage_error(y, y_pred)
        print("RMSE: {}".format(rmse))
        print("MAPE: {}".format(mape))

        self._data_processor.write_model_field(self.model_id, 'rsme', rmse)
        self._data_processor.write_model_field(self.model_id, 'mape', mape)

        self._data_processor.write_inner_model(self.model_id, self._inner_model, use_pickle=True)

        return rmse

    def predict(self, inputs, get_graph=False, graph_data=None):

        data = pd.DataFrame(inputs)
        additional_data = {'x_indicators': self.x_indicators,
                           'y_indicators': self.y_indicators,
                           'periods': self.periods,
                           'organisations': self.organisations,
                           'scenarios': self.scenarios,
                           'x_columns': self.x_columns,
                           'y_columns': self.y_columns}

        x, x_pd, x_columns = self._data_processor.get_x_for_prediction(data, additional_data)

        x_scaler = self._get_scaler(retrofit=True)
        x_sc = x_scaler.transform(x)

        inner_model = self._get_inner_model(for_prediction=True)

        y_sc = inner_model.predict(x_sc)
        y_scaler = self._get_scaler(retrofit=True, is_out=True)
        y = y_scaler.inverse_transform(y_sc)

        data = x_pd.copy()
        data[self.y_columns] = y

        graph_bin = None

        if get_graph:
            graph_bin = self._get_graph_bin(data, graph_data)

        outputs = data.drop(self.x_columns, axis=1)

        indicators_description = {x_ind: {'indicator': self._data_processor.get_indicator_name(x_ind),
                                          'report_type': self._data_processor.get_indicator_report_type(x_ind)}
                                  for x_ind in self.x_indicators + self.y_indicators}

        return outputs.to_dict('records'), indicators_description, graph_bin

    def calculate_feature_importances(self, **kwargs):

        inner_model = self._get_inner_model(for_prediction=True)

        fi = pd.DataFrame(inner_model.coef_[0], columns=['feature_importance'])

        fi['feature'] = self.x_columns
        fi = fi.sort_values(by='feature_importance', ascending=False)
        fi['indicator'] = fi['feature'].apply(self._data_processor.get_indicator_name)
        fi['report_type'] = fi['feature'].apply(self._data_processor.get_indicator_report_type)

        fi = fi.to_dict('records')
        self._data_processor.write_feature_importances(self.model_id, fi)

        return fi

    def _get_scaler(self, retrofit=False, is_out=False):

        if retrofit:
            scaler = self._data_processor.read_scaler(self.model_id, is_out)
            if not scaler:
                scaler = StandardScaler()
        else:
            scaler = StandardScaler()

        return scaler

    def _get_inner_model(self, for_prediction=False):

        if for_prediction:
            inner_model = self._data_processor.read_inner_model(self.model_id, use_pickle=True)

            if not inner_model:
                raise ProcessorException('model id={} is not found'.format(self.model_id))
        else:
            inner_model = self._create_inner_model()

        self._inner_model = inner_model

        return inner_model

    def _get_model_for_feature_importances(self):
        return self._get_inner_model()

    @staticmethod
    def _create_inner_model():

        model = LinearRegression()# tol=.0001, eta0=.01)

        return model


class DataProcessor:

    def __init__(self):

        self._db_connector = DB_CONNECTOR

    def load_data(self, raw_data, overwrite=False):

        pd_data = pd.DataFrame(raw_data)

        pd_indicators = pd_data.groupby(['indicator_name', 'indicator_id', 'report_type'], as_index=False).count()
        indicators_list = pd_indicators[['indicator_name', 'indicator_id', 'report_type']].to_dict(orient='records')
        for line in indicators_list:
            self._db_connector.write_indicator(line['indicator_id'], line['indicator_name'], line['report_type'])

        pd_analytics = pd_data.groupby(['analytics_1_type',
                                        'analytics_1_name', 'analytics_1_id'], as_index=False).count()

        analytics_list = pd_analytics[['analytics_1_type',
                                        'analytics_1_name', 'analytics_1_id']].to_dict(orient='records')

        for line in analytics_list:
            self._db_connector.write_analytics(line['analytics_1_id'],
                                               line['analytics_1_name'], line['analytics_1_type'])

        self._db_connector.write_raw_data(raw_data, overwrite=overwrite)

    def read_model_description_from_db(self, model_id):
        return self._db_connector.read_model_description(model_id)

    def get_indicator_ids(self, indicators):

        if not indicators:
            return []

        result = [self._db_connector.read_indicator_from_name_type(ind_line['indicator'],
                                                                   ind_line['report_type'])['indicator_id']
                  for ind_line in indicators]
        return result

    def get_indicators_data_from_parameters(self, indicator_parameters):
        result = []
        for parameters_line in indicator_parameters:
            result_line = self._db_connector.read_indicator_from_id(parameters_line['id'])
            result_line.update(parameters_line)
            result.append(result_line)

        return result

    def get_indicator_description_from_id(self, indicator_id):

        result = self._db_connector.read_indicator_from_id(indicator_id)

        return result

    def get_indicator_short_id(self, indicator_id):
        result = ''
        indicator_line = self._db_connector.read_indicator_from_id(indicator_id)
        if indicator_line:
            result = indicator_line['short_id']

        return result

    def get_indicator_name(self, indicator_id):
        indicator_id = indicator_id.replace('_value', '')
        result = None
        indicator_line = self._db_connector.read_indicator_from_id(indicator_id)
        if indicator_line:
            result = indicator_line['indicator']

        return result

    def get_indicator_report_type(self, indicator_id):
        indicator_id = indicator_id.replace('_value', '')
        result = None
        indicator_line = self._db_connector.read_indicator_from_id(indicator_id)
        if indicator_line:
            result = indicator_line['report_type']

        return result

    def read_raw_data(self, indicators=None, date_from=None):
        raw_data = self._db_connector.read_raw_data(indicators, date_from)
        return raw_data

    @staticmethod
    def get_additional_data(raw_data):
        pd_data = pd.DataFrame(raw_data)
        organisations, scenarios, periods = list(pd_data['organisation'].unique()),\
                                            list(pd_data['scenario'].unique()), \
                                            list(pd_data['period'].unique())
        return organisations, scenarios, periods

    def write_model_to_db(self, model_description):
        self._db_connector.write_model_description(model_description)

    def get_x_y_for_fitting(self, data, additional_data, encode_fields=None, model_filter=None):

        data = pd.DataFrame(data)
        if model_filter:
            for filter_field, filter_value in model_filter.items():
                data = data.loc[data[filter_field]==filter_value].copy()
        data = self._add_short_ids_to_data(data)
        data_grouped, data_grouped_values = self._prepare_dataset_group(data)

        indicators = additional_data['x_indicators'] + additional_data['y_indicators']

        data = self._prepare_dataset_add_indicators_analytics(data_grouped, data_grouped_values, indicators)

        additional_data['years'] = list(set([self._get_year(period) for period in additional_data['periods']]))
        additional_data['months'] = list(set([self._get_month(period) for period in additional_data['periods']]))

        if encode_fields:
            data = self._prepare_dataset_one_hot_encode(data, additional_data, encode_fields)

        data = self._drop_non_numeric_columns(data, indicators)

        data = self._process_na(data, additional_data['y_indicators'])

        y_columns = ['ind_{}'.format(ind_line['short_id']) for ind_line in additional_data['y_indicators']]

        inputs = data.copy()
        inputs = inputs.drop(y_columns, axis=1)

        outputs = data.copy()
        outputs = outputs[y_columns]

        x_columns = list(inputs.columns)
        x = inputs.to_numpy()

        y_columns = list(outputs.columns)
        y = outputs.to_numpy()

        return x, y, x_columns, y_columns

    def get_x_for_prediction(self, data, additional_data, encode_fields=None):

        data = pd.DataFrame(data)

        data = self._add_short_ids_to_data(data)
        data_grouped, data_grouped_values = self._prepare_dataset_group(data)

        indicators = additional_data['x_indicators']
        x_columns = additional_data['x_columns']

        data = self._prepare_dataset_add_columns_for_prediction(data_grouped, data_grouped_values, indicators, x_columns)

        additional_data['years'] = list(set([self._get_year(period) for period in additional_data['periods']]))
        additional_data['months'] = list(set([self._get_month(period) for period in additional_data['periods']]))

        if encode_fields:
            data = self._prepare_dataset_one_hot_encode(data, additional_data, encode_fields)

        data = self._process_na(data)

        data_num = self._drop_non_numeric_columns(data, indicators)

        x = data_num.to_numpy()

        return x, data

    def write_columns(self, model_id, x_columns, y_columns):
        self._db_connector.write_model_columns(model_id, x_columns, y_columns)

    def write_scaler(self, model_id, scaler, is_out=False):
        scaler_packed = pickle.dumps(scaler, protocol=pickle.HIGHEST_PROTOCOL)
        self._db_connector.write_model_scaler(model_id, scaler_packed, is_out)

    def write_feature_importances(self, model_id, feature_importances):
        self._db_connector.write_model_fi(model_id, feature_importances)

    def read_feature_importances(self, model_id):
        model_description = self.read_model_description_from_db(model_id)
        return model_description.get('feature_importances')

    def read_scaler(self, model_id, is_out=False):
        model_description = self.read_model_description_from_db(model_id)
        scaler_name = 'y_scaler' if is_out else 'x_scaler'
        scaler_packed = model_description[scaler_name]
        return pickle.loads(scaler_packed)

    def write_inner_model(self, model_id, inner_model, use_pickle=False):
        if use_pickle:
            model_packed = pickle.dumps(inner_model, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            if not os.path.isdir('tmp'):
                os.mkdir('tmp')

            inner_model.save('tmp/model')

            zipf = zipfile.ZipFile('tmp/model.zip', 'w', zipfile.ZIP_DEFLATED)
            self._zipdir('tmp/model', zipf)
            zipf.close()

            with open('tmp/model.zip', 'rb') as f:
                model_packed = f.read()

        self._db_connector.write_inner_model(model_id, model_packed)

        if not use_pickle:
            os.remove('tmp/model.zip')
            shutil.rmtree('tmp/model')

    def read_inner_model(self, model_id, use_pickle=False):

        if use_pickle:
            model_description = self.read_model_description_from_db(model_id)
            inner_model = pickle.loads(model_description['inner_model'])
        else:

            if not os.path.isdir('tmp'):
                os.mkdir('tmp')

            model_description = self.read_model_description_from_db(model_id)
            inner_model = model_description['inner_model']
            with open('tmp/model.zip', 'wb') as f:
                f.write(inner_model)

            with zipfile.ZipFile('tmp/model.zip', 'r') as zip_h:
                zip_h.extractall('tmp/model')

            inner_model = keras.models.load_model('tmp/model')

            if not use_pickle:
                os.remove('tmp/model.zip')
                shutil.rmtree('tmp/model')

        return inner_model

    def write_model_field(self, model_id, field_name, value):
        self._db_connector.write_model_field(model_id, field_name, value)

    def read_model_field(self, model_id, field_name):
        model_description = self.read_model_description_from_db(model_id)
        value = model_description.get(field_name)
        return value

    def _add_short_ids_to_data(self, dataset):

        dataset['indicator_short_id'] = dataset['indicator_id'].apply(self._get_indicator_short_id)
        dataset['analytics_short_id'] = dataset[['analytics_1_id',
                                                 'analytics_1_type']].apply(self._get_analytics_short_id, axis=1)

        return dataset

    @staticmethod
    def _prepare_dataset_group(dataset):

        columns_to_drop = ['indicator_name', 'indicator_id', 'report_type', 'analytics_1_id', 'analytics_1_name',
                           'analytics_1_type']
        if '_id' in list(dataset.columns):
            columns_to_drop.append('_id')

        dataset.drop(columns_to_drop, axis=1, inplace=True)
        dataset.rename({'indicator_short_id': 'indicator'}, axis=1, inplace=True)
        dataset.rename({'analytics_short_id': 'analytics'}, axis=1, inplace=True)

        data_grouped_values = dataset.groupby(['indicator', 'analytics', 'organisation', 'scenario', 'period',
                                               'periodicity'],
                                              as_index=False)
        data_grouped_values = data_grouped_values.sum()
        data_grouped_values = data_grouped_values[['indicator', 'analytics', 'organisation', 'scenario', 'period',
                                                   'periodicity', 'value']]

        data_grouped = dataset.groupby(['organisation', 'scenario', 'period', 'periodicity'], as_index=False).max()
        data_grouped = data_grouped[['organisation', 'scenario', 'period', 'periodicity']]

        return data_grouped, data_grouped_values

    def _prepare_dataset_add_indicators_analytics(self, dataset, dataset_grouped_values, indicators):

        data_pr = dataset.copy()
        data_pr = self._add_shifting_periods_to_data(data_pr, indicators)

        for ind_line in indicators:
            period_shift = ind_line.get('period_shift') or 0
            if period_shift:
                period_column = 'period_' + ('m{}'.format(-period_shift) if period_shift < 0
                                             else 'p{}'.format(period_shift))
            else:
                period_column = 'period'

            with_analytics = ind_line.get('use_analytics')

            data_str = dataset_grouped_values.loc[(dataset_grouped_values['indicator'] == ind_line['short_id'])]
            if with_analytics:
                c_analytics = list(data_str['analytics'].unique())
                if '' in c_analytics:
                    c_analytics.pop(c_analytics.index(''))
            else:
                c_analytics = ['']

            for an_el in c_analytics:

                data_str_a = data_str.loc[(data_str['analytics'] == an_el)]

                data_str_a = data_str_a.groupby(['organisation', 'scenario', 'period'], as_index=False).sum()
                if period_column != 'period':
                    data_str_a = data_str_a.rename({'period': period_column}, axis=1)

                data_str_a = data_str_a[['organisation', 'scenario', period_column, 'value']]

                data_pr = data_pr.merge(data_str_a, on=['organisation', 'scenario', period_column], how='left')

                column_name = 'ind_{}'.format(ind_line['short_id'])

                if with_analytics:
                    column_name = '{}_an_{}'.format(column_name, an_el)

                if period_shift:
                    column_name = '{}_p_'.format(column_name) + ('m{}'.format(-period_shift) if period_shift < 0
                                                                 else 'p{}'.format(period_shift))

                data_pr = data_pr.rename({'value': column_name}, axis=1)

        data_pr = self._add_month_year_to_data(data_pr)
        return data_pr

    def _prepare_dataset_add_columns_for_prediction(self, dataset, dataset_grouped_values, indicators, columns):

        data_pr = dataset.copy()
        data_pr = self._add_shifting_periods_to_data(data_pr, indicators)

        for column in columns:

            col_list = column.split('_')
            with_analytics = False
            period_shift = 0
            ind = ''
            an = ''
            if len(col_list) == 2:
                ind = col_list[1]
            elif len(col_list) == 4:
                ind = col_list[1]
                if col_list[2] == 'an':
                    with_analytics = True
                    an = col_list[3]
                else:
                    period_shift = col_list[3]
                    if period_shift[0] == 'p':
                        period_shift = int(period_shift[1:])
                    else:
                        period_shift = -int(period_shift[1:])
            else:
                ind = col_list[1]
                an = col_list[3]
                period_shift = col_list[5]
                if period_shift[0] == 'p':
                    period_shift = int(period_shift[1:])
                else:
                    period_shift = -int(period_shift[1:])

            if period_shift:
                period_column = 'period_' + ('m{}'.format(-period_shift) if period_shift < 0
                                             else 'p{}'.format(period_shift))
            else:
                period_column = 'period'

            data_str_a = dataset_grouped_values.loc[(dataset_grouped_values['indicator'] == ind)
                                                    & (dataset_grouped_values['analytics'] == an)]

            if period_column != 'period':
                data_str_a = data_str_a.rename({'period': period_column}, axis=1)

            data_str_a = data_str_a[['organisation', 'scenario', period_column, 'value']]

            data_pr = data_pr.merge(data_str_a, on=['organisation', 'scenario', period_column], how='left')

            data_pr = data_pr.rename({'value': column}, axis=1)

        data_pr = self._add_month_year_to_data(data_pr)
        return data_pr

    def _prepare_dataset_one_hot_encode(self, dataset, additional_data, encode_fields):

        fields_dict = {encode_1: additional_data[encode_2] for encode_1, encode_2 in encode_fields.items()}

        for field_name, field_values in fields_dict.items():
            dataset = self._one_hot_encode(dataset, field_name, field_values)

        return dataset

    def _add_month_year_to_data(self, dataset):
        dataset['month'] = dataset['period'].apply(self._get_month)
        dataset['year'] = dataset['period'].apply(self._get_year)

        return dataset

    def _add_shifting_periods_to_data(self, dataset, indicators):

        period_numbers = [ind_line.get('period_shift') for ind_line in indicators if ind_line.get('period_shift')]
        period_columns = []
        dataset['shift'] = 0
        for period_num in period_numbers:

            column_name = 'period_' + ('p{}'.format(period_num) if period_num > 0 else 'm{}'.format(-period_num))
            period_columns.append(column_name)

            dataset['shift'] = period_num
            dataset[column_name] = dataset[['period', 'shift', 'periodicity']].apply(self._get_shifting_period, axis=1)

        dataset = dataset.drop(['shift'], axis=1)
        return dataset

    @staticmethod
    def _drop_non_numeric_columns(dataset, indicators):

        columns_to_drop = ['organisation', 'scenario', 'period', 'periodicity', 'year', 'month']
        period_numbers = [ind_line.get('period_shift') for ind_line in indicators if ind_line.get('period_shift')]
        period_columns = []
        for period_num in period_numbers:
            column_name = 'period_' + ('p{}'.format(period_num) if period_num > 0 else 'm{}'.format(-period_num))
            period_columns.append(column_name)

        columns_to_drop = columns_to_drop + period_columns

        dataset = dataset.drop(columns_to_drop, axis=1)

        return dataset

    def _process_na(self, dataset, y_indicators=None):

        if y_indicators:
            y_columns = ['ind_{}'.format(ind_line['short_id']) for ind_line in y_indicators]

            dataset['y_na'] = dataset[y_columns].apply(self._get_na, axis=1)

            dataset = dataset.loc[dataset['y_na'] == False]

            dataset = dataset.drop(['y_na'], axis=1)

        dataset = dataset.fillna(0)

        return dataset

    @staticmethod
    def _get_na(data_line):
        is_na = False
        for el in data_line:
            is_na = math.isnan(el)
            if is_na:
                break

        not_el = True
        if not is_na:
            for el in data_line:
                if el:
                    not_el = False
                    break

            is_na = not_el


        return is_na

    @staticmethod
    def _get_shifting_period(period_data):

        period = period_data['period']
        shift = period_data['shift']
        periodicity = period_data['periodicity']

        day, month, year = map(int, period.split('.'))

        if periodicity in ['day', 'week', 'decade']:

            period_date = datetime.datetime(year, month, day)

            day_shift = shift
            if periodicity == 'week':
                day_shift = shift*7
            elif periodicity == 'decade':
                day_shift = shift*10

            period_date = period_date + datetime.timedelta(days=day_shift)

            day = period_date.day
            month = period_date.month
            year = period_date.year

        else:

            month_shift = shift
            if periodicity == 'quarter':
                month_shift = shift*3
            elif periodicity == 'half_year':
                month_shift = shift*6
            elif periodicity == 'nine_months':
                month_shift = shift*9
            elif periodicity == 'year':
                month_shift = shift*12

            month_shift_ = shift % (12 if shift > 0 else -12)
            year_shift_ = shift // (12 if shift > 0 else -12)

            month += month_shift_
            year += year_shift_

            if month < 1:
                month += 12
                year -=1
            elif month > 12:
                month -=12
                year +=1

        return '{:02}.{:02}.{}'.format(day, month, year)

    def _get_indicator_short_id(self, indicator_id):
        return self._db_connector.get_short_id(indicator_id)

    def _get_analytics_short_id(self, analytics_data):
        return self._db_connector.get_short_id(analytics_data[0] + ' ' + analytics_data[1])

    @staticmethod
    def _get_year(date_str):
        return int(date_str.split('.')[2])

    @staticmethod
    def _get_month(date_str):
        return int(date_str.split('.')[1])

    @staticmethod
    def _one_hot_encode(dataset, field_name, indicators):
        for indicator in indicators:
            dataset[field_name.lower() + ' '
                    + str(indicator)] = dataset[field_name].apply(lambda x: 1 if x == indicator else 0)

        return dataset

    @staticmethod
    def _zipdir(path, ziph):
        # ziph is zipfile handle
        for root, dirs, files in os.walk(path):
            c_dir = root
            c_dir = 'tmp/' + c_dir[10:]

            for file in files:
                ziph.write(os.path.join(root, file),
                           os.path.relpath(os.path.join(c_dir, file),
                                           os.path.join(path, '..')))

    def _get_indicator_id(self, indicator, report_type):

        indicator_from_db = self._db_connector.read_indicator_from_name_type(indicator, report_type)

        if not indicator_from_db:
            indicator_id = 'ind_' + settings_controller.get_id()
            self._db_connector.write_indicator(indicator_id, indicator, report_type)
        else:
            indicator_id = indicator_from_db['indicator_id']

        return indicator_id

    def _get_indicator_id_one_arg(self, indicator_descr):
        return self._get_indicator_id(indicator_descr[0], indicator_descr[1])


@JobProcessor.job_processing
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

    processor = ModelProcessor(parameters)
    prediction, indicator_description, graph_bin = processor.predict(parameters)

    result = dict(status='OK',
                  error_text='',
                  result=prediction,
                  indicator_description=indicator_description,
                  description='model predicted')

    if graph_bin:
        result['graph_data'] = base64.b64encode(graph_bin).decode(encoding='utf-8')
    return result


@JobProcessor.job_processing
def calculate_feature_importances(parameters):

    processor = ModelProcessor(parameters)
    fi = processor.calculate_feature_importances(parameters)

    result = dict(status='OK', error_text='', result=fi, description='model feature importances calculated')

    return result


def get_feature_importances(parameters):

    processor = ModelProcessor(parameters)
    get_graph = parameters.get('get_graph')
    fi, graph_bin = processor.get_feature_importances(parameters)

    result = dict(status='OK', error_text='', result=fi, description='model feature importances recieved')
    if get_graph:
        result['graph_data'] = base64.b64encode(graph_bin).decode(encoding='utf-8')
    return result


def get_rsme(parameters):

    processor = ModelProcessor(parameters)
    rsme, mspe = processor.get_rsme(parameters)

    result = dict(status='OK', error_text='', rsme=rsme, mspe=mspe, description='model rsme recieved')

    return result


def set_db_connector(parameters):
    global DB_CONNECTOR
    if not DB_CONNECTOR:
        DB_CONNECTOR = db_connector.Connector(parameters, initialize=True)
