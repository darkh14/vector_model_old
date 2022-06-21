import settings_controller
from logger import ProcessorException as ProcessorException
import db_connector

import numpy as np
import pandas as pd
import os
import math
from abc import ABCMeta, abstractmethod
import json
import hashlib

from job_processor import JobProcessor
from data_loader import LoadingProcessor

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
import eli5

from eli5.sklearn import PermutationImportance
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import tensorflow as tf

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
        self._db_connector = DB_CONNECTOR
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

        job_id = parameters.get('job_id') or ''

        history = self.model.fit(epochs=parameters.get('epochs'),
                                 validation_split=parameters.get('validation_split'),
                                 retrofit=retrofit,
                                 date_from=date_from, job_id=job_id)

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
                                                                          graph_data=graph_data,
                                                                          additional_parameters=parameters)

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

        result, graph_bin = self.model.get_feature_importances(parameters.get('get_graph'), parameters.get('extended'))

        return result, graph_bin

    def get_rsme(self, parameters):

        model_description = parameters.get('model')

        if not model_description:
            raise ProcessorException('model is not in parameters')

        self.model = self._get_model(model_description)

        rsme, mspe = self.model.get_rsme()

        return rsme, mspe

    def get_model_parameters(self, parameters):

        model_description = parameters.get('model')

        if not model_description:
            raise ProcessorException('model is not in parameters')

        self.model = self._get_model(model_description)

        model_parameters = self.model.get_model_parameters()

        return model_parameters

    def get_factor_analysis_data(self, parameters):

        model_description = parameters.get('model')

        if not model_description:
            raise ProcessorException('model is not in parameters')

        self.model = self._get_model(model_description)

        inputs = parameters.get('inputs')

        if not inputs:
            raise ProcessorException('inputs is not in parameters')

        output_indicator_id = parameters.get('output_indicator_id')

        if not output_indicator_id:
            raise ProcessorException('output indicator id is not in parameters')

        result, indicator_description, graph_data = self.model.get_factor_analysis_data(inputs,
                                                     output_indicator_id,
                                                     step=parameters.get('step'),
                                                     get_graph=parameters.get('get_graph'))

        return result, indicator_description, graph_data

    def _get_model(self, model_description):

        need_to_update = bool(model_description.get('need_to_update'))

        model_type = model_description.get('type')
        if not model_type:
            model_type = self._data_processor.read_model_field(model_description['id'], 'type')
            model_description['type'] = model_type

        if not model_type:
            raise ProcessorException('model type not in model description')

        model_class = None
        if model_type == 'neural_network':
            model_class = NeuralNetworkModel
        elif model_type == 'linear_regression':
            model_class = LinearModel
        elif model_type == 'periodic_neural_network':
            model_class = PeriodicNeuralNetworkModel

        if not model_class:
            raise ProcessorException('model type "{}" is not supported'.format(model_type))

        model_description['need_to_update'] = need_to_update

        model = model_class(model_description['id'], model_description)

        return model


class BaseModel:

    __metaclass__ = ABCMeta
    type = ''

    def __init__(self, model_id, model_parameters):

        name = model_parameters.get('name') or ''

        x_indicators = model_parameters.get('x_indicators')
        y_indicators = model_parameters.get('y_indicators')
        need_to_update = model_parameters.get('need_to_update')
        model_filter = model_parameters.get('filter')

        self.model_id = model_id

        self._db_connector = DB_CONNECTOR
        self._data_processor = DataProcessor()

        self.is_fit = False
        self.fitting_date = None
        self.fitting_is_started = False
        self.fitting_start_date = None
        self.fitting_job_id = ''

        self._field_to_update = ['name', 'type', 'is_fit', 'fitting_is_started', 'fitting_start_date',  'fitting_date',
                                 'filter', 'x_indicators', 'y_indicators', 'periods', 'organisations',
                                 'scenarios', 'x_columns', 'y_columns', 'x_analytics', 'y_analytics',
                                 'x_analytic_keys', 'y_analytic_keys', 'feature_importances', 'fitting_job_id']

        description_from_db = self._data_processor.read_model_description_from_db(self.model_id)

        if description_from_db:
            for field in self._field_to_update:
                setattr(self, field, description_from_db.get(field))
        else:
            for field in self._field_to_update:
                setattr(self, field, [])
            self.name = name
            self.filter = model_filter

            self.is_fit = False
            self.fitting_date = None
            self.fitting_is_started = False
            self.fitting_start_date = None
            self.fitting_job_id = ''

        if x_indicators:
            self.x_indicators = self._data_processor.get_indicators_data_from_parameters(x_indicators)

        if y_indicators:
            self.y_indicators = self._data_processor.get_indicators_data_from_parameters(y_indicators)

        self.type = model_parameters['type']

        self._inner_model = None
        self._retrofit = False

        if model_filter:
            self.filter = model_filter

        self.need_to_update = not description_from_db or need_to_update
        self.graph_file_name = 'graph.png'
        self.graph_fi_file_name = 'fi_graph.png'
        self.graph_fa_file_name = 'fa_graph.png'

    def update_model(self, data):
        if self.need_to_update:
            organisations, scenarios, periods = self._data_processor.get_additional_data(data)
            self.periods = periods
            self.organisations = organisations
            self.scenarios = scenarios

            model_description = {field: getattr(self, field) for field in self._field_to_update}

            self._data_processor.write_model_to_db(self.model_id, model_description)
            self.need_to_update = False

    def fit(self, epochs=100, validation_split=0.2, retrofit=False, date_from=None, job_id=''):

        job_id = job_id or ''

        self.fitting_is_started = True
        self.fitting_date = None
        self.fitting_start_date = datetime.datetime.now()
        self.fitting_job_id = job_id
        self._data_processor.write_model_field(self.model_id, 'fitting_is_started', self.fitting_is_started)
        self._data_processor.write_model_field(self.model_id, 'fitting_date', self.fitting_date)
        self._data_processor.write_model_field(self.model_id, 'fitting_start_date', self.fitting_start_date)
        self._data_processor.write_model_field(self.model_id, 'fitting_job_id', self.fitting_job_id)

        self.fit_model(epochs=epochs, validation_split=validation_split, retrofit=retrofit, date_from=date_from)

        self.is_fit = True
        self.fitting_date = datetime.datetime.now()
        self.fitting_is_started = False
        self.fitting_start_date = None
        self._data_processor.write_model_field(self.model_id, 'fitting_is_started', self.fitting_is_started)
        self._data_processor.write_model_field(self.model_id, 'is_fit', self.is_fit)
        self._data_processor.write_model_field(self.model_id, 'fitting_date', self.fitting_date)
        self._data_processor.write_model_field(self.model_id, 'fitting_start_date', self.fitting_start_date)
        self._data_processor.write_model_field(self.model_id, 'fitting_job_id', '')

    @abstractmethod
    def fit_model(self, epochs=100, validation_split=0.2, retrofit=False, date_from=None):
        """method for fitting model"""

    @abstractmethod
    def predict(self, inputs, get_graph=False, graph_data=None, additional_parameters=None):
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

        fi = fi.to_dict('records')
        self._data_processor.write_feature_importances(self.model_id, fi)

        return fi

    def get_feature_importances(self, get_graph=False, extended=False):
        fi = self._data_processor.read_feature_importances(self.model_id)
        if not fi:
            raise ProcessorException('Feature importances is not calculated')

        if not extended:
            fi_pd = pd.DataFrame(fi)
            fi_pd['count'] = 1
            fi_pd['ind'] = fi_pd.index

            fi_pd_group = fi_pd.groupby(['indicator_short_id'], as_index=False).sum()
            fi_pd_group['feature_importance'] = fi_pd['feature_importance']/fi_pd['count']

            fi_pd_ind = fi_pd[['indicator_short_id', 'ind']].groupby(['indicator_short_id'], as_index=False).min()

            fi_pd_group = fi_pd_group.drop('ind', axis=1)
            fi_pd_group = fi_pd_group.merge(fi_pd_ind, on='indicator_short_id', how='inner')
            fi_pd_group = fi_pd_group.merge(fi_pd[['ind', 'indicator']], on='ind', how='inner')

            fi_pd_group = fi_pd_group.sort_values(by='feature_importance', ascending=False)
            fi = fi_pd_group.to_dict('records')

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

    def get_model_parameters(self):
        rsme = self._data_processor.read_model_field(self.model_id, 'rsme')
        mspe = self._data_processor.read_model_field(self.model_id, 'mspe')

        is_fit = self._data_processor.read_model_field(self.model_id, 'is_fit')
        fitting_is_started = self._data_processor.read_model_field(self.model_id, 'fitting_is_started')
        fitting_date = self._data_processor.read_model_field(self.model_id, 'fitting_date')

        if fitting_date:
            fitting_date = fitting_date.strftime('%d.%m.%Y %H:%M:%S')

        fitting_start_date = self._data_processor.read_model_field(self.model_id, 'fitting_start_date')
        if fitting_start_date:
            fitting_start_date = fitting_start_date.strftime('%d.%m.%Y %H:%M:%S')

        fitting_job_id = self._data_processor.read_model_field(self.model_id, 'fitting_job_id')

        model_parameters = {'rsme': rsme, 'mspe': mspe, 'is_fit': is_fit, 'fitting_date': fitting_date,
                            'fitting_is_started': fitting_is_started, 'fitting_start_date': fitting_start_date,
                            'fitting_job_id': fitting_job_id}

        return model_parameters

    @abstractmethod
    def get_factor_analysis_data(self, inputs, output_indicator_id, step=0.3, get_graph=False):
        """method for getting factor analysis data"""

    def _check_data(self, data, additional_parameters=None):
        """method for checking data when predicting, ex. indicators in data and in model accordance"""
        pass

    def _get_scaler(self, retrofit=False, is_out=False):

        if retrofit:
            scaler = self._data_processor.read_scaler(self.model_id, is_out)
            if not scaler:
                scaler = StandardScaler()
        else:
            scaler = StandardScaler()

        return scaler

    def _get_graph_bin(self, data, graph_data):

        x_graph, y_graph = self._get_dataframe_for_graph(data, graph_data['x_indicator'],
                                                         graph_data['y_indicator'], graph_data.get('periods'))

        x_indicator_descr = self._db_connector.read_indicator_from_type_id(graph_data['x_indicator']['type'],
                                                                           graph_data['x_indicator']['id'])

        x_label = self._get_indicator_label_text(x_indicator_descr)

        y_indicator_descr = self._db_connector.read_indicator_from_type_id(graph_data['y_indicator']['type'],
                                                                           graph_data['y_indicator']['id'])

        y_label = self._get_indicator_label_text(y_indicator_descr)

        self._make_graph(x_graph, y_graph, x_label, y_label)

        graph_bin = self._read_graph_file()

        return graph_bin

    @staticmethod
    def _get_indicator_label_text(indicator):
        result = indicator.get('name') or indicator.get('id')
        if indicator.get('report_type'):
            result = indicator['report_type'] + '\n' + result

        return result

    def _make_graph(self, x, y, x_label, y_label):

        x_max = max(x.max(), -(x.min()))
        x_mul = math.floor(math.log10(x_max)) if x_max else 0
        x_mul = math.floor(x_mul/3)*3
        x_mul = max(x_mul, 0)

        x = x*10**(-x_mul)

        y_max = max(y.max(), -(y.min()))
        y_mul = math.floor(math.log10(y_max)) if y_max else 0
        y_mul = math.floor(y_mul/3)*3
        y_mul = max(y_mul, 0)

        y = y*10**(-y_mul)

        fig, ax = plt.subplots()

        ax.plot(x, y) # , label='y_test')

        ax.set_xlabel(x_label + ('\n' + '\\ {}'.format(10**x_mul) if x_mul else ''))
        ax.set_ylabel(y_label + ('\n' + '\\ {}'.format(10**y_mul) if y_mul else ''))
        # ax.legend()

        fig.set_figwidth(8)  # ширина и
        fig.set_figheight(8)  # высота "Figure"

        ax.grid()

        # plt.show()
        fig.savefig(self.graph_file_name)

    def _read_graph_file(self, graph_type='main'):

        graph_file_name = ''
        if graph_type == 'main':
            graph_file_name = self.graph_file_name
        elif graph_type == 'fi':
            graph_file_name = self.graph_fi_file_name
        elif graph_type == 'fa':
            graph_file_name = self.graph_fa_file_name

        if not graph_file_name:
            raise ProcessorException('Graph type {} is not allowed'.format(graph_file_name))

        f = open(graph_file_name, 'rb')
        result = f.read()
        f.close()

        return result

    def _get_dataframe_for_graph(self, data, x_indicator, y_indicator, periods=None):

        x_indicator_descr = self._db_connector.read_indicator_from_type_id(x_indicator['type'], x_indicator['id'])

        x_columns = []
        for col in self.x_columns:
            if col == 'month':
                continue

            col_list = col.split('_')
            if len(col_list) == 6:
                continue

            if col_list[1] == x_indicator_descr['short_id']:
                x_columns.append(col)

        y_indicator_descr = self._db_connector.read_indicator_from_type_id(y_indicator['type'], y_indicator['id'])
        y_columns = []
        for col in self.y_columns:
            col_list = col.split('_')
            if len(col_list)==6:
                continue

            if col_list[1] == y_indicator_descr['short_id']:
                y_columns.append(col)

        if periods:
            data = data.loc[data['period'].isin(periods)].copy()

        data = data[x_columns + y_columns].copy()

        data['x'] = data[x_columns].apply(sum, axis=1)
        data['x'] = data['x'].apply(round)

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

        graph_bin = self._read_graph_file(graph_type='fi')

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

    def _get_data_for_fa_graph(self, outputs, indicators_description, output_indicator_id, steps):

        result = outputs.copy()
        result['count'] = 1
        result_group = result.groupby(by=['indicator_id', 'koef'], as_index=False).sum()

        result_group['reverse_delta'] = -result_group['max_delta']
        result_group = result_group.sort_values(by=['reverse_delta', 'koef'])

        num_columns = list(result_group.columns)
        num_columns = [col for col in num_columns if col not in ['indicator_id', 'koef', 'count']]

        for col in num_columns:
            result_group[col] = result_group[col]/result_group['count']

        result_null = result_group.loc[result_group['koef'] == 0]
        result_null = result_null[['indicator_id', 'out_value']]
        result_null = result_null.rename({'out_value': 'out_value_null'}, axis=1)

        result_group = result_group.merge(result_null, on=['indicator_id'], how='left')
        result_group = result_group.drop(['count', 'reverse_delta'], axis=1)

        result_group['value_percent'] = result_group[['out_value', 'out_value_null']].apply(lambda x:
                                                                        100*x[0]/x[1] if x[1] else 0, axis=1)

        result_group['indicator'] = result_group['indicator_id'].apply(self._get_indicator_name_from_description)

        return result_group

    def _get_indicator_name_from_description(self, indicator_id):
        descr_lines = list(filter(lambda x: x['id'] == indicator_id, self.x_indicators + self.y_indicators))
        return descr_lines[0]['name']

    def _make_fa_graph(self, dataset, steps):

        indicator_descr = dataset[['indicator', 'indicator_id']].groupby(by=['indicator', 'indicator_id'],
                                                                         as_index=False).max()

        indicators = list(indicator_descr['indicator'].values)

        ind_list = [el.replace(' ', '\n') for el in indicators]

        x0 = dataset['indicator_id'].unique()

        fig, ax = plt.subplots()

        x_shifts = [(el-len(steps)/2)/7 for el in list(range(len(steps)))]
        ind = 0
        x_ticks_is_set = False

        for step in steps:
            y = dataset.loc[dataset['koef'] == step]['value_percent'].to_numpy()
            x = [(el + x_shifts[ind]) for el in list(range(len(x0)))]
            ax.bar(x, y, width=0.05*len(steps) /(0.5*len(x0)), align='edge', label="{:.0f}".format(step*100) + ' %')
            if ind > len(steps)/2 - 1 and not x_ticks_is_set:
                ax.set_xticks(x)
                x_ticks_is_set = True
                ax.set_xticklabels(ind_list)
            ind += 1

        # ax.set_xlabel(indicators)

        ax.set_facecolor('seashell')
        fig.set_facecolor('floralwhite')
        fig.set_figwidth(20)  # ширина Figure
        fig.set_figheight(10)  # высота Figure

        ax.set_ylabel('Изменение выходного показателя, %')

        ax.set_title('Факторный анализ')

        # ax.set_ylim([0, 110])
        # ax.set_xlim([-0.5, len(x) + 1])

        ax.legend(loc=(1, 0.5), title='Изменение входных\n показателей, %')

        for rect in ax.patches:

            y_value = rect.get_y() + rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2

            label = "{:.0f}".format(rect.get_height())

            ax.annotate(
                label,  # Use `label` as label
                (x_value, y_value),  # Place label at end of the bar
                xytext=(0, 0),  # Vertically shift label by `space`
                textcoords="offset points",  # Interpret `xytext` as offset in points
                ha='center',  # Horizontally center label
                va='bottom',
                fontsize=6)  # Vertically align label differently for
            previous_label = rect.get_height()

        fig.savefig(self.graph_fa_file_name)

    def _get_fa_graph_bin(self, values, steps):

        self._make_fa_graph(values, steps)

        graph_bin = self._read_graph_file(graph_type='fa')

        return graph_bin


class NeuralNetworkModel(BaseModel):

    type = 'neural_network'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._epochs = 0
        self._validation_split = 0.2

        self._temp_input = None

    def fit_model(self, epochs=100, validation_split=0.2, retrofit=False, date_from=None):

        x, y = self._prepare_for_fit(retrofit, date_from)

        inner_model = self._get_inner_model(x.shape[1], y.shape[1], retrofit=retrofit)

        self._inner_model = inner_model
        self._epochs = epochs or 1000
        self._validation_split = validation_split or 0.2

        inner_model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='MeanSquaredError',
                            metrics=['RootMeanSquaredError'])

        history = inner_model.fit(x, y, epochs=self._epochs, verbose=2, validation_split=self._validation_split)

        self._inner_model = inner_model

        self._write_after_fit(x, y)

        return history.history

    def predict(self, inputs, get_graph=False, graph_data=None, additional_parameters=None):

        data = pd.DataFrame(inputs)

        additional_data = {'model_id': self.model_id,
                           'x_indicators': self.x_indicators,
                           'y_indicators': self.y_indicators,
                           'periods': self.periods,
                           'organisations': self.organisations,
                           'scenarios': self.scenarios,
                           'x_columns': self.x_columns,
                           'y_columns': self.y_columns}

        result, errors = self._check_data(inputs)

        if not result:
            errors = [error + '\n' for error in errors]
            raise ProcessorException(''.join(errors))

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

        description = {'x_indicators': self.x_indicators,
                       'y_indicators': self.y_indicators,
                       'x_analytics': self.x_analytics,
                       'y_analytics': self.y_analytics}

        return outputs.to_dict('records'), description, graph_bin

    def calculate_feature_importances(self, date_from=None, epochs=1000, retrofit=False, validation_split=0.2):

        if not retrofit:
            date_from = None
        else:
            date_from = datetime.datetime.strptime(date_from, '%d.%m.%Y')

        indicator_filter = [ind_data['short_id'] for ind_data in self.x_indicators + self.y_indicators]

        data = self._data_processor.read_raw_data(indicator_filter, date_from)
        additional_data = {'x_indicators': self.x_indicators,
                           'y_indicators': self.y_indicators,
                           'periods': self.periods,
                           'organisations': self.organisations,
                           'scenarios': self.scenarios,
                           'x_columns': self.x_columns,
                           'y_columns': self.y_columns}
        x, y, x_columns, y_columns = self._data_processor.get_x_y_for_fitting(data, additional_data)

        self._temp_input = x
        # self._inner_model = self._get_inner_model(len(self.x_columns), len(self.y_columns), retrofit=retrofit)

        epochs = epochs or 1000
        validation_split = validation_split or 0.2

        fi_model = KerasRegressor(build_fn=self._get_model_for_feature_importances,
                                  epochs=epochs,
                                  verbose=2,
                                  validation_split=validation_split)
        fi_model.fit(x, y)

        fi = self._calculate_fi_from_model(fi_model, x, y, x_columns)

        return fi

    def _prepare_for_fit(self, retrofit, date_from, add_params=None):
        if not retrofit:
            date_from = None
        else:
            date_from = datetime.datetime.strptime(date_from, '%d.%m.%Y')

        indicator_filter = [ind_data['short_id'] for ind_data in self.x_indicators + self.y_indicators]

        data = self._data_processor.read_raw_data(indicator_filter, date_from=date_from, ad_filter=self.filter)
        if retrofit and self.need_to_update:
            raise ProcessorException('Model can not be updated when retrofit')

        self.update_model(data)

        additional_data = {'model_id': self.model_id,
                           'x_indicators': self.x_indicators,
                           'y_indicators': self.y_indicators,
                           'periods': self.periods,
                           'organisations': self.organisations,
                           'scenarios': self.scenarios,
                           'x_columns': self.x_columns,
                           'y_columns': self.y_columns}
        if add_params:
            additional_data.update(add_params)

        # encode_fields = {'organisation': 'organisations', 'year': 'years', 'month': 'months'}
        encode_fields = None
        x, y, x_columns, y_columns = self._data_processor.get_x_y_for_fitting(data, additional_data, encode_fields)
        x_analytics, y_analytics, x_analytic_keys, y_analytic_keys = \
            self._data_processor.get_analytics_description(x_columns + y_columns, self.x_indicators, self.y_indicators)
        self.x_columns = x_columns
        self.y_columns = y_columns

        self.x_analytics = x_analytics
        self.y_analytics = y_analytics

        self.x_analytic_keys = x_analytic_keys
        self.y_analytic_keys = y_analytic_keys

        if not retrofit:
            self._data_processor.write_columns(self.model_id, x_columns, y_columns)
            self._data_processor.write_analytics_decription(self.model_id, x_analytics, y_analytics, x_analytic_keys, y_analytic_keys)

        return x, y

    def _write_after_fit(self, x, y):

        y_pred = self._inner_model.predict(x)

        rmse = self._calculate_rsme(y, y_pred)
        mspe = self._calculate_mspe(y, y_pred)
        print("RMSE: {}".format(rmse))
        print("MSPE: {}".format(mspe))

        self._data_processor.write_model_field(self.model_id, 'rsme', rmse)
        self._data_processor.write_model_field(self.model_id, 'mspe', mspe)

        self._data_processor.write_inner_model(self.model_id, self._inner_model)

    def _calculate_fi_from_model(self, fi_model, x, y, x_columns):
        perm = PermutationImportance(fi_model, random_state=42).fit(x, y)

        fi = pd.DataFrame(perm.feature_importances_, columns=['feature_importance'])
        fi['feature'] = x_columns
        fi = fi.sort_values(by='feature_importance', ascending=False)

        fi = fi.loc[fi['feature']!='month'].copy()

        fi[['indicator_short_id', 'indicator']] = fi[['feature']].apply(self._data_processor.get_indicator_data_from_fi,
                                                                        axis=1, result_type='expand')
        fi[['analytic_key_id', 'analytics']] = fi[['feature']].apply(self._data_processor.get_analytics_data_from_fi,
                                                                     axis=1, result_type='expand')

        fi = fi.to_dict('records')
        self._data_processor.write_feature_importances(self.model_id, fi)

        return fi

    @staticmethod
    def _calculate_rsme(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

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
        # model_copy = clone_model(self._inner_model) #
        model_copy = self._create_inner_model(len(self.x_columns), len(self.y_columns))
        model_copy.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='MeanSquaredError',
                           metrics=['RootMeanSquaredError'])
        return model_copy

    def get_factor_analysis_data(self, inputs, output_indicator_id, step=0.3, get_graph=False):

        data = pd.DataFrame(inputs)
        data = self._data_processor.add_short_ids_to_raw_data(data)

        additional_data = {'x_indicators': self.x_indicators,
                           'y_indicators': self.y_indicators,
                           'periods': self.periods,
                           'organisations': self.organisations,
                           'scenarios': self.scenarios,
                           'x_columns': self.x_columns + self.y_columns,
                           'y_columns': self.y_columns}

        indicators_description = self.x_indicators + self.y_indicators

        period_columns = list()
        for indicator_data in self.x_indicators:
            if indicator_data['period_shift']:
                column_name = 'period_{}{}'.format('m' if indicator_data['period_shift'] < 0 else 'p',
                                                   -indicator_data['period_shift'] if indicator_data['period_shift'] < 0
                                                   else indicator_data['period_shift'])

                if column_name not in period_columns:
                    period_columns.append(column_name)

        inner_model = self._get_inner_model(retrofit=True)

        step = step or 0.3

        output_col_list = list(filter(lambda dt: dt['id'] == output_indicator_id, self.y_indicators))

        if not output_col_list:
            raise ProcessorException('Output indicator id not in model')

        output_column_name = 'ind_' + output_col_list[0]['short_id']

        result = pd.DataFrame()

        multistep = False
        if isinstance(step, list) or isinstance(step, tuple):
            steps = step
            if 0 not in steps:
                steps.append(0)
            multistep = True
        else:
            steps = [-step, 0, step]

        steps.sort()

        for indicator_data in self.x_indicators:

            if indicator_data['period_shift']:
                continue

            cur_data = pd.DataFrame()

            for step in steps:

                raw_cur_data = data.copy()

                if step:
                    cur_data_ind = raw_cur_data.loc[raw_cur_data['indicator_short_id']==indicator_data['short_id']].copy()
                    cur_data_ind['value'] = cur_data_ind['value']*(1 + step)
                    raw_cur_data.loc[raw_cur_data['indicator_short_id'] == indicator_data['short_id']] = cur_data_ind

                encode_fields = None
                x, x_y_pd = self._data_processor.get_x_for_prediction(raw_cur_data, additional_data, encode_fields)

                columns_to_drop = self.y_columns + ['organisation', 'scenario', 'period', 'periodicity',
                                                    'year'] + period_columns

                x = x_y_pd.drop(columns_to_drop, axis=1).to_numpy()

                y = inner_model.predict(x)

                y_pd = pd.DataFrame(y, columns=self.y_columns)
                y = y_pd[[output_column_name]].to_numpy()

                x_y_pd[output_column_name] = y

                input_indicator_columns = list()
                if indicator_data['use_analytics']:
                    for col_name in self.x_columns:

                        if col_name=='month':
                            continue

                        col_list = col_name.split('_')
                        if col_list[1] == indicator_data['short_id'] and len(col_list) == 4:
                            input_indicator_columns.append(col_name)
                else:
                    col_name = 'ind_' + indicator_data['short_id']
                    if indicator_data.get('period_number'):
                        col_name += '_p_f{}'.format(indicator_data['period_number'])
                    input_indicator_columns.append(col_name)

                if not len(cur_data):
                    cur_data = x_y_pd.copy()
                    cur_data['indicator_id'] = indicator_data['id']
                    columns_to_drop = ['organisation', 'scenario', 'periodicity', 'year',
                                       'month'] + period_columns + self.x_columns + self.y_columns
                    cur_data = cur_data.drop(columns_to_drop, axis=1)

                cur_data['koef'] = step
                cur_data['out_value'] = x_y_pd[output_column_name]
                cur_data['in_value'] = x_y_pd[input_indicator_columns].apply(sum, axis=1)

                if not len(result):
                    result = cur_data.copy()
                else:
                    result = pd.concat([result, cur_data], axis=0)

        result_min = result.loc[result['koef'] == steps[0]].groupby('indicator_id', as_index=False).sum()
        result_min = result_min.rename({'out_value': 'out_value_min'}, axis=1)
        result_min = result_min.drop(['koef', 'in_value'], axis=1)
        result_max = result.loc[result['koef'] == steps[-1]].groupby('indicator_id', as_index=False).sum()
        result_max = result_max.rename({'out_value': 'out_value_max'}, axis=1)
        result_max = result_max.drop(['koef', 'in_value'], axis=1)

        result = result.merge(result_min, on=['indicator_id'], how='left')
        result = result.merge(result_max, on=['indicator_id'], how='left')
        result['max_delta'] = abs(result['out_value_max'] - result['out_value_min'])

        result = result.sort_values(by='max_delta', ascending=False)

        result = result.drop(['out_value_min'], axis=1)
        result = result.drop(['out_value_max'], axis=1)

        graph_bin = None
        if get_graph:
            graph_data = self._get_data_for_fa_graph(result, indicators_description, output_indicator_id, steps)
            graph_bin = self._get_fa_graph_bin(graph_data, steps)

        result = result.drop(['max_delta'], axis=1)
        output = result.to_dict(orient='records')

        return output, indicators_description, graph_bin

    def _check_data(self, data, additional_parameters=None):

        data = pd.DataFrame(data)
        data = self._data_processor.add_short_ids_to_raw_data(data)

        errors = []

        if not self.is_fit:
            raise ProcessorException('Model is not fit. Prediction is impossible!')

        # # check indicators
        # indicator_short_ids_from_data = data['indicator_short_id'].unique()
        # for ind in self.x_indicators:
        #     if ind['short_id'] not in indicator_short_ids_from_data:
        #         errors.append('indicator {} not found in input data'.format(ind['name']))
        #
        # # check analytic keys
        # analytic_keys_short_ids_from_data = data['analytics_key_id'].unique()
        # for el in self.x_analytic_keys:
        #     if el['short_id'] not in analytic_keys_short_ids_from_data:
        #         errors.append('analytic key id {} not found in input data'.format(el['id']))

        return not errors, errors

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


class PeriodicNeuralNetworkModel(NeuralNetworkModel):

    type = 'periodic_neural_network'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_processor = PeriodicDataProcessor()

        model_parameters = args[1]
        model_description = self._data_processor.read_model_description_from_db(model_parameters['id'])
        if model_description:
            self.past_history = model_description.get('past_history')
            self.future_target = model_description.get('future_target')
        else:
            self.past_history = model_parameters.get('past_history')
            self.future_target = model_parameters.get('future_target')

    def update_model(self, data):
        super().update_model(data)
        model_description = {'past_history': self.past_history, 'future_target': self.future_target}
        self._data_processor.write_model_to_db(self.model_id, model_description)

    def fit_model(self, epochs=100, validation_split=0.2, retrofit=False, date_from=None):

        add_parameters = {'past_history': self.past_history,  'future_target': self.future_target}
        x, y = self._prepare_for_fit(retrofit, date_from, add_parameters)

        disable_gpu()

        BATCH_SIZE = 256
        BUFFER_SIZE = 10000

        x_y_data = tf.data.Dataset.from_tensor_slices((x, y))
        x_y_data = x_y_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

        inner_model = self._get_inner_model(x.shape[-2:], y.shape[-2], retrofit=retrofit)

        self._epochs = epochs or 10
        self._validation_split = validation_split or 0.2

        inner_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mae',
                            metrics=['RootMeanSquaredError'])

        history = inner_model.fit(x_y_data, epochs=self._epochs, steps_per_epoch=50, verbose=2)

        self._inner_model = inner_model

        self._write_after_fit(x, y.reshape(y.shape[0], y.shape[1]))

        return history.history

    def predict(self, inputs, get_graph=False, graph_data=None, additional_parameters=None):

        past_periods = additional_parameters['past_periods']
        future_periods = additional_parameters['future_periods']

        if not past_periods:
            raise ProcessorException('"past_periods" not in parameters')

        if not future_periods:
            raise ProcessorException('"future_periods" not in parameters')

        result, errors = self._check_data(inputs, additional_parameters)

        if not result:
            errors = [error + '\n' for error in errors]
            raise ProcessorException(''.join(errors))

        data = pd.DataFrame(inputs)

        scaler = self._data_processor.read_scaler(self.model_id)
        additional_data = {'model_id': self.model_id,
                           'scaler': scaler,
                           'x_indicators': self.x_indicators,
                           'y_indicators': self.y_indicators,
                           'periods': self.periods,
                           'organisations': self.organisations,
                           'scenarios': self.scenarios,
                           'x_columns': self.x_columns,
                           'y_columns': self.y_columns,
                           'past_history': self.past_history,
                           'future_target': self.future_target,
                           'past_periods': past_periods,
                           'future_periods': future_periods}

        encode_fields = None
        x, x_pd = self._data_processor.get_x_for_prediction(data, additional_data, encode_fields)

        disable_gpu()

        inner_model = self._get_inner_model(retrofit=True)

        y_sc = inner_model.predict(x)

        x_columns_ind = [el for el in self.x_columns if el != 'month']

        y_add = np.zeros((y_sc.shape[1], len(x_columns_ind)))

        indexes = list()
        for ind in range(len(x_columns_ind)):
            if x_columns_ind[ind] in self.y_columns:
                y_ind = self.y_columns.index(x_columns_ind[ind])
                y_add[:, ind] = y_sc[y_ind]
                indexes.append(ind)

        y_add = scaler.inverse_transform(y_add)
        y = list()
        for ind in indexes:
            y.append(y_add[:, ind])

        y = np.array(y)
        # y = y.flatten().tolist()

        graph_bin = None

        if get_graph:
            graph_bin = self._get_graph_bin(y[0], graph_data)

        description = {'x_indicators': self.x_indicators,
                       'y_indicators': self.y_indicators,
                       'x_analytics': self.x_analytics,
                       'y_analytics': self.y_analytics}

        y = y[0].tolist()

        return y, description, graph_bin

    @staticmethod
    def _create_inner_model(inputs_number=0, outputs_number=0):
        multi_step_model = tf.keras.models.Sequential()
        multi_step_model.add(tf.keras.layers.LSTM(300,
                                                  return_sequences=True,
                                                  input_shape=inputs_number,
                                                  name='ltsm_1'))
        multi_step_model.add(tf.keras.layers.LSTM(200, activation='relu', name='ltsm_2'))
        multi_step_model.add(tf.keras.layers.Dense(outputs_number, name='dense_3'))

        return multi_step_model

    @staticmethod
    def _calculate_rsme(y_true, y_pred):
        return np.sqrt(np.nanmean(np.square(y_true**2 - y_pred**2)))

    def _check_data(self, data, additional_parameters=None):

        result, errors = super()._check_data(data, additional_parameters)

        data = pd.DataFrame(data)

        periods = list(data['period'].unique())
        if len(periods) < len(additional_parameters['past_periods']):
            errors.append('Nodel needs {} periods to predict. '
                          'There are only {} in inputs'.format(additional_parameters['past_periods'], len(periods)))
            result = False

        return result, errors

    @staticmethod
    def _calculate_mspe(y_true, y_pred):

        eps = np.zeros(y_true.shape)
        eps[::] = 0.0001
        y_p = np.c_[abs(np.expand_dims(y_true, axis=2)), abs(np.expand_dims(y_pred, axis=2)),
                    np.expand_dims(eps, axis=2)]
        y_p = np.max(y_p, axis=2)

        return np.sqrt(np.nanmean(np.square(((y_true - y_pred) / y_p))))

    def _get_graph_bin(self, data, graph_data):

        x_pred, y_pred, x_val, y_val = self._get_dataframe_for_graph_periods(data, graph_data['validation_data'],
                                                                 graph_data['periods'])

        self._make_graph(x_pred, y_pred, x_val, y_val)

        graph_bin = self._read_graph_file()

        return graph_bin

    @staticmethod
    def _get_dataframe_for_graph_periods(y, y_val, periods):

        if len(periods) != len(y):
            raise ProcessorException('Number of periods ({}) not equal to '
                                     'number of predicted values ({})'.format(len(periods), len(y)))

        pd_data = pd.DataFrame(periods, columns=['period'])
        pd_data['y_pred'] = y

        pd_data_val = pd.DataFrame(y_val)

        if len(y_val):
            pd_data = pd_data.merge(pd_data_val, on='period', how='left')
            pd_data = pd_data.rename({'value': 'y_val'}, axis=1)
            pd_data = pd_data.fillna('na')
        else:
            pd_data['y_val'] = 'na'

        data_pred = pd_data[['period', 'y_pred']]
        data_val = pd_data.loc[pd_data['y_val'] != 'na']

        return data_pred['period'].to_numpy(), data_pred['y_pred'].to_numpy(), \
               data_val['period'].to_numpy(), data_val['y_val'].to_numpy()

    def _make_graph(self, x_pred, y_pred, x_val, y_val):

        if len(y_val):
            y_val_max = y_val.max()
            y_val_min = y_val.min()
        else:
            y_val_max = 0
            y_val_min = 0

        y_max = max(y_pred.max(), -(y_pred.min()), y_val_max, -(y_val_min))

        y_mul = math.floor(math.log10(y_max))
        y_mul = math.floor(y_mul/3)*3
        y_mul = max(y_mul, 0)

        y_pred_m = y_pred*10**(-y_mul)

        if len(y_val):
            y_val_m = y_val*10**(-y_mul)

        fig, ax = plt.subplots()

        ax.plot(x_pred, y_pred_m, label='Спрогнозированные значения')
        if len(y_val):
            ax.plot(x_val, y_val_m, label='Проверочные значения')

        # ax.set_xlabel(x_label + '\n' + '\\ {}'.format(10**x_mul))
        ax.set_ylabel('* {}'.format(10**y_mul))

        ax.legend()

        fig.set_figwidth(8)  # ширина и
        fig.set_figheight(8)  # высота "Figure"

        plt.xticks(rotation=45)

        ax.grid()

        # plt.show()
        fig.savefig(self.graph_file_name)


class LinearModel(BaseModel):

    type = 'linear_regression'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit_model(self, **kwargs):

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

    def predict(self, inputs, get_graph=False, graph_data=None, additional_parameters=None):

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

        indicators_description = {x_ind: {'indicator': self._data_processor.get_indicator_name(x_ind)}
                                  for x_ind in self.x_indicators + self.y_indicators}

        return outputs.to_dict('records'), indicators_description, graph_bin

    def calculate_feature_importances(self, **kwargs):

        inner_model = self._get_inner_model(for_prediction=True)

        fi = pd.DataFrame(inner_model.coef_[0], columns=['feature_importance'])

        fi['feature'] = self.x_columns
        fi = fi.sort_values(by='feature_importance', ascending=False)
        fi['indicator'] = fi['feature'].apply(self._data_processor.get_indicator_name)

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

    def _check_data(self, data):
        pass

    def get_factor_analysis_data(self, inputs, output_indicator_id, step=0.3, get_graph=False):
        return None, None, None


class DataProcessor:

    def __init__(self):

        self._db_connector = DB_CONNECTOR
        self._id_processor = IdProcessor()

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
            result_line = self._db_connector.read_indicator_from_type_id(parameters_line['type'], parameters_line['id'])
            if not result_line:
                raise ProcessorException('indicator {}, id {} not found in indicators'.format(parameters_line.get('name'), parameters_line['id']))
            result_line.update(parameters_line)
            result.append(result_line)

        return result

    def get_indicator_description_from_id(self, indicator_id):

        result = self._db_connector.read_indicator_from_type_id(indicator_id)

        return result

    def get_indicator_short_id(self, indicator_id):
        result = ''
        indicator_line = self._db_connector.read_indicator_from_id(indicator_id)
        if indicator_line:
            result = indicator_line['short_id']

        return result

    def get_indicator_data_from_fi(self, column_name):

        short_id = column_name['feature'].split('_')[1]

        result = '', '', ''
        indicator_line = self._db_connector.read_indicator_from_short_id(short_id)

        return short_id, indicator_line

    def get_analytics_data_from_fi(self, column_name):

        column_list = column_name['feature'].split('_')

        short_id = ''
        if len(column_list) == 4:
            if column_list[2] == 'an':
                short_id = column_list[3]
        elif len(column_list) == 6:
            short_id = column_list[3]

        analytics_line = self._db_connector.read_analytics_from_key_id(short_id)

        return short_id, analytics_line

    def read_raw_data(self, indicators=None, date_from=None, ad_filter=None):
        raw_data = self._db_connector.read_raw_data(indicators, date_from, ad_filter)
        return raw_data

    @staticmethod
    def get_additional_data(raw_data):
        pd_data = pd.DataFrame(raw_data)
        organisations, scenarios, periods = list(pd_data['organisation'].unique()),\
                                            list(pd_data['scenario'].unique()), \
                                            list(pd_data['period'].unique())
        return organisations, scenarios, periods

    def write_model_to_db(self, model_id, model_description):
        model_description['model_id'] = model_id
        self._db_connector.write_model_description(model_description)

    def get_x_y_for_fitting(self, data, additional_data, encode_fields=None):

        data = self.get_data_for_fitting(data, additional_data, encode_fields=encode_fields)

        data = self._drop_non_numeric_columns(data, additional_data['x_indicators'] + additional_data['y_indicators'])

        # y_columns = ['ind_{}'.format(ind_line['short_id']) for ind_line in additional_data['y_indicators']]
        y_columns = self._get_columns_by_indicators(data.columns, additional_data['y_indicators'])

        inputs = data.copy()
        inputs = inputs.drop(y_columns, axis=1)

        outputs = data.copy()
        outputs = outputs[y_columns]

        x_columns = list(inputs.columns)
        x = inputs.to_numpy()

        y_columns = list(outputs.columns)
        y = outputs.to_numpy()

        return x, y, x_columns, y_columns

    def get_data_for_fitting(self, data, additional_data, encode_fields=None):

        data = pd.DataFrame(data)

        data = self._add_month_year_to_data(data)

        data_grouped, data_grouped_values = self._prepare_dataset_group(data)

        indicators = additional_data['x_indicators'] + additional_data['y_indicators']

        data = self._prepare_dataset_add_indicators_analytics(data_grouped, data_grouped_values, indicators)

        additional_data['years'] = list(set([self._get_year(period) for period in additional_data['periods']]))
        additional_data['months'] = list(set([self._get_month(period) for period in additional_data['periods']]))

        if encode_fields:
            data = self._prepare_dataset_one_hot_encode(data, additional_data, encode_fields)

        data = self._process_na(data, additional_data)

        return data

    @staticmethod
    def _get_columns_by_indicators(columns, indicators):
        ind_ids = [ind['short_id'] for ind in indicators]

        result_columns = [col for col in columns if len(col) >= 10 and col[4:11] in ind_ids
                          and len(col.split('_')) != 6]

        return result_columns

    def get_x_for_prediction(self, data, additional_data, encode_fields=None):

        data = pd.DataFrame(data)

        data = self.add_short_ids_to_raw_data(data)
        data = self._add_month_year_to_data(data)

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

    def write_analytics_decription(self, model_id, x_analytics, y_analytics, x_analytic_keys, y_analytic_keys):
        self._db_connector.write_model_analytics(model_id, x_analytics, y_analytics, x_analytic_keys, y_analytic_keys)

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

    @staticmethod
    def _prepare_dataset_group(dataset):

        columns_to_drop = ['indicator', 'analytics']
        if '_id' in list(dataset.columns):
            columns_to_drop.append('_id')

        if 'report_type' in list(dataset.columns):
            columns_to_drop.append('report_type')

        dataset.drop(columns_to_drop, axis=1, inplace=True)
        dataset.rename({'indicator_short_id': 'indicator'}, axis=1, inplace=True)
        dataset.rename({'analytics_key_id': 'analytics'}, axis=1, inplace=True)

        data_grouped_values = dataset.groupby(['indicator', 'analytics', 'organisation', 'scenario', 'period',
                                               'periodicity', 'month', 'year'],
                                              as_index=False)
        data_grouped_values = data_grouped_values.sum()
        data_grouped_values = data_grouped_values[['indicator', 'analytics', 'organisation', 'scenario', 'period',
                                                   'periodicity', 'month', 'year', 'value']]

        data_grouped = dataset.groupby(['organisation', 'scenario', 'period', 'periodicity', 'month', 'year'],
                                       as_index=False).max()
        data_grouped = data_grouped[['organisation', 'scenario', 'period', 'periodicity', 'month', 'year']]

        return data_grouped, data_grouped_values

    def _prepare_dataset_add_indicators_analytics(self, dataset, dataset_grouped_values, indicators):

        data_pr = dataset.copy()
        data_pr = self._add_shifting_periods_to_data(data_pr, indicators)

        for ind_line in indicators:
            period_shift = ind_line.get('period_shift') or 0
            period_number = ind_line.get('period_number') or 0
            period_accumulation = ind_line.get('period_accumulation') or 0
            if period_shift:
                period_column = 'period_' + ('m{}'.format(-period_shift) if period_shift < 0
                                             else 'p{}'.format(period_shift))
            elif period_number:
                period_column = 'year'
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

                data_str_a = data_str_a.groupby(['organisation', 'scenario', 'period', 'month', 'year'], as_index=False).sum()

                if period_shift:
                    data_str_a = data_str_a.drop(['year'], axis=1)
                elif period_number:
                    data_str_a = data_str_a.loc[data_str_a['month'] == period_number].copy()
                    data_str_a = data_str_a.drop(['period'], axis=1)

                data_str_a = data_str_a.rename({'period': period_column}, axis=1)

                data_str_a = data_str_a[['organisation', 'scenario', period_column, 'value']]

                data_pr = data_pr.merge(data_str_a, on=['organisation', 'scenario', period_column], how='left')

                column_name = 'ind_{}'.format(ind_line['short_id'])

                if with_analytics:
                    column_name = '{}_an_{}'.format(column_name, an_el)

                if period_shift:
                    column_name = '{}_p_'.format(column_name) + ('m{}'.format(-period_shift) if period_shift < 0
                                                                 else 'p{}'.format(period_shift))
                elif period_number:
                    column_name = '{}_p_f{}'.format(column_name, period_number)

                data_pr = data_pr.rename({'value': column_name}, axis=1)

        return data_pr

    def _prepare_dataset_add_columns_for_prediction(self, dataset, dataset_grouped_values, indicators, columns):

        data_pr = dataset.copy()
        data_pr = self._add_shifting_periods_to_data(data_pr, indicators)

        for column in columns:

            if column == 'month':
                continue

            col_list = column.split('_')
            with_analytics = False
            period_shift = 0
            period_number = 0
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
                    period_value = col_list[3]

                    if period_value[0] == 'p':
                        period_shift = int(period_value[1:])
                    elif period_value[0] == 'm':
                        period_shift = -int(period_value[1:])
                    elif period_value[0] == 'f':
                        period_number = int(period_value[1:])
            else:
                ind = col_list[1]
                an = col_list[3]
                period_value = col_list[5]
                if period_value[0] == 'p':
                    period_shift = int(period_value[1:])
                elif period_value[0] == 'm':
                    period_shift = -int(period_value[1:])
                elif period_value[0] == 'f':
                    period_number = int(period_value[1:])

            if period_shift:
                period_column = 'period_' + ('m{}'.format(-period_shift) if period_shift < 0
                                             else 'p{}'.format(period_shift))
            elif period_number:
                period_column = 'year'
            else:
                period_column = 'period'

            data_str_a = dataset_grouped_values.loc[(dataset_grouped_values['indicator'] == ind)
                                                    & (dataset_grouped_values['analytics'] == an)]

            if period_shift:
                data_str_a = data_str_a.drop(['year'], axis=1)
            elif period_number:
                data_str_a = data_str_a.loc[data_str_a['month'] == period_number].copy()
                data_str_a = data_str_a.drop(['period'], axis=1)

            data_str_a = data_str_a.rename({'period': period_column}, axis=1)

            data_str_a = data_str_a[['organisation', 'scenario', period_column, 'value']]

            data_pr = data_pr.merge(data_str_a, on=['organisation', 'scenario', period_column], how='left')

            data_pr = data_pr.rename({'value': column}, axis=1)

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

    def add_short_ids_to_raw_data(self, raw_data):
        return self._id_processor.add_short_ids_to_raw_data(raw_data)
    #
    # def _make_short_id_from_list(self, list_value):
    #     if list_value:
    #         short_id_list = [el['short_id'] for el in list_value]
    #         short_id_list.sort()
    #         str_val = ''.join(short_id_list)
    #         return self.get_hash(str_val)
    #     else:
    #         return ''
    #
    # def _make_short_id_from_dict(self, dict_value):
    #     str_val = dict_value['id'] + dict_value.get('type') or ''
    #     return self.get_hash(str_val)
    #
    # def _add_short_id_to_analytics(self, analytics_list):
    #     for an_el in analytics_list:
    #         an_el['short_id'] = self._make_short_id_from_dict(an_el)
    #     return analytics_list
    #
    # @staticmethod
    # def get_hash(value):
    #     if not value.replace(' ', ''):
    #         return ''
    #     data_hash = hashlib.md5(value.encode())
    #     return data_hash.hexdigest()[-7:]

    @staticmethod
    def _drop_non_numeric_columns(dataset, indicators):

        columns_to_drop = ['organisation', 'scenario', 'period', 'periodicity', 'year']
        period_numbers = [ind_line.get('period_shift') for ind_line in indicators if ind_line.get('period_shift')]
        period_columns = []
        for period_num in period_numbers:
            column_name = 'period_' + ('p{}'.format(period_num) if period_num > 0 else 'm{}'.format(-period_num))
            period_columns.append(column_name)

        columns_to_drop = columns_to_drop + period_columns

        dataset = dataset.drop(columns_to_drop, axis=1)

        return dataset

    def _process_na(self, dataset, additional_data=None):

        if additional_data:

            all_columns = list(dataset.columns)
            period_columns = [col for col in all_columns if col.find('period_') != -1]
            non_digit_columns = ['organisation', 'scenario', 'period',
                                 'year', 'periodicity'] + period_columns

            digit_columns = [_el for _el in all_columns if _el not in non_digit_columns]

            x_short_ids = [_el['short_id'] for _el in additional_data['x_indicators']]
            y_short_ids = [_el['short_id'] for _el in additional_data['y_indicators']]

            x_digit_columns = []
            x_digit_without_month = []
            y_digit_columns = []

            for col in digit_columns:

                if col == 'month':
                    x_digit_columns.append(col)
                else:

                    col_list = col.split('_')

                    if col_list[1] in x_short_ids:
                        x_digit_columns.append(col)
                        x_digit_without_month.append(col)

                    if col_list[1] in y_short_ids:
                        y_digit_columns.append(col)

            dataset['x_not_del'] = dataset[x_digit_without_month].any(axis=1)
            dataset['y_not_del'] = dataset[y_digit_columns].any(axis=1)
            dataset['not_del'] = dataset[['x_not_del', 'y_not_del']].apply(lambda x: x[0] and x[1], axis=1)

            dataset = dataset.loc[dataset['not_del'] == True].copy()

            col_to_delete = []

            for col in x_digit_without_month:
                if not dataset[col].any():
                    col_to_delete.append(col)

            columns_not_to_del = [_el for _el in all_columns if _el not in col_to_delete]

            dataset = dataset[columns_not_to_del].copy()

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

    def _get_indicator_short_id(self, indicator_type, indicator_id):
        value_str = indicator_id + indicator_type
        return self._db_connector.get_short_id(value_str)

    def _get_analytics_short_id(self, analytics_data):
        return self._db_connector.get_short_id(analytics_data[0] + ' ' + analytics_data[1])

    def _get_analytics_description_from_short_id(self, short_id):
        return self._db_connector.read_analytics_from_short_id(short_id)

    def _get_analytics_description_from_key_id(self, key_id):
        return self._db_connector.read_analytics_from_key_id(key_id)

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

    def get_analytics_description(self, columns, x_indicators, y_indicators):

        x_ind_ids = [ind['short_id'] for ind in x_indicators]
        y_ind_ids = [ind['short_id'] for ind in y_indicators]

        x_analytics = []
        y_analytics = []

        x_analytic_keys = []
        y_analytic_keys = []

        for column in columns:
            if len(column) > 4:
                if column[4:11] in x_ind_ids:
                    c_col_list = column.split('_')
                    if len(c_col_list) >= 3 and c_col_list[2] == 'an':
                        short_id = c_col_list[3]
                        analytics = self._get_analytics_description_from_key_id(short_id)
                        x_analytic_keys.append({'short_id': short_id, 'analytics': analytics})

                if column[4:11] in y_ind_ids:
                    c_col_list = column.split('_')
                    if len(c_col_list) >= 3 and c_col_list[2] == 'an':
                        short_id = c_col_list[3]
                        analytics = self._get_analytics_description_from_key_id(short_id)
                        y_analytic_keys.append({'short_id': short_id, 'analytics': analytics})

        x_an_ids = []
        for key_val in x_analytic_keys:
            for an_el in key_val['analytics']:
                if an_el['short_id'] not in x_an_ids:
                    x_an_ids.append(an_el['short_id'])
                    x_analytics.append(self._get_analytics_description_from_short_id(an_el['short_id']))

        y_an_ids = []
        for key_val in y_analytic_keys:
            for an_el in key_val['analytics']:
                if an_el['short_id'] not in y_an_ids:
                    y_an_ids.append(an_el['short_id'])
                    y_analytics.append(self._get_analytics_description_from_short_id(an_el['short_id']))

        return x_analytics, y_analytics, x_analytic_keys, y_analytic_keys


class IdProcessor(LoadingProcessor):

    def __init__(self):
        pass


class PeriodicDataProcessor(DataProcessor):

    def get_x_y_for_fitting(self, data, additional_data, encode_fields=None):

        data = self.get_data_for_fitting(data, additional_data, encode_fields=encode_fields)

        data['period_dt'] = data['period'].apply(self.get_period)

        x_columns = [el for el in list(data.columns) if el[:4] == 'ind_' or el == 'month']
        x_columns_ind = [el for el in x_columns if el != 'month']
        y_columns = ['ind_{}'.format(ind_line['short_id']) for ind_line in additional_data['y_indicators']]

        data_sc = data.copy()

        x_sc = data_sc[x_columns_ind].to_numpy()

        scaler = StandardScaler()
        scaler.fit(x_sc)

        self.write_scaler(additional_data['model_id'], scaler)

        x_sc = scaler.transform(x_sc)

        data_sc[x_columns_ind] = x_sc

        organisations = data['organisation'].unique()
        scenarios = data['scenario'].unique()

        x, y = np.array(list()), np.array(list())

        for organisation in organisations:

            for scenario in scenarios:

                data_el = data_sc.loc[(data_sc['organisation']==organisation) & (data_sc['scenario']==scenario)].copy()
                data_el = data_el.sort_values('period_dt')

                x_data = data_el[x_columns].values
                y_data = data_el[y_columns].values

                if data_el.shape[0] >= additional_data['past_history'] + additional_data['future_target']:
                    x_sc, y_sc = self.multivariate_data(x_data, y_data, 0,
                                                        data_el.shape[0] - additional_data['future_target'],
                                                        additional_data['past_history'],
                                                        additional_data['future_target'], 1)
                    if x.any():
                        x = np.concatenate((x, x_sc), axis=0)
                    else:
                        x = x_sc

                    if y.any():
                        y = np.concatenate((y, y_sc), axis=0)
                    else:
                        y = y_sc

        return x, y, x_columns, y_columns

    @staticmethod
    def multivariate_data(dataset, target, start_index, end_index, history_size,
                          target_size, step, single_step=False):
        data = []
        labels = []

        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size

        for i in range(start_index, end_index):
            indices = range(i - history_size, i, step)

            data.append(dataset[indices])

            if target.any():
                if single_step:
                    labels.append(target[i + target_size])
                else:
                    labels.append(target[i:i + target_size])

        return np.array(data), np.array(labels)

    @staticmethod
    def get_period(period_str):
        return datetime.datetime.strptime(period_str, '%d.%m.%Y')

    def get_x_for_prediction(self, data, additional_data, encode_fields=None):

        data = self.add_short_ids_to_raw_data(data)
        data = self.get_data_for_fitting(data, additional_data, encode_fields=encode_fields)

        data = data.loc[data['period'].isin(additional_data['past_periods'])].copy()

        data['period_dt'] = data['period'].apply(self.get_period)

        organisations = data['organisation'].unique()

        if len(organisations) != 1:
            raise ProcessorException('Periodic model needs data with one organisation for prediction,' +
                                     'but now data contains {} organisations'.format(organisations))

        scenarios = data['scenario'].unique()

        if len(scenarios) != 1:
            raise ProcessorException('Periodic model needs data with one scenario for prediction,' +
                                     'but now data contains {} scenarios'.format(scenarios))

        data = data.sort_values('period_dt')

        x_columns_ind = [el for el in additional_data['x_columns'] if el != 'month']

        data_sc = data.copy()

        x_sc = data_sc[x_columns_ind]
        x_sc = additional_data['scaler'].transform(x_sc)

        data_sc[x_columns_ind] = x_sc

        x_data = data_sc[additional_data['x_columns']].values

        x, y = self.multivariate_data(x_data, np.array(list()), 0,
                                      data.shape[0]+1,
                                      additional_data['past_history'],
                                      0, 1)

        return x, data

    def _process_na(self, dataset, y_indicators=None):

        dataset = dataset.fillna(0)

        return dataset


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

    result = dict(status='OK', error_text='', result=fi, description='model feature importances is recieved')
    if get_graph:
        result['graph_data'] = base64.b64encode(graph_bin).decode(encoding='utf-8')
    return result


def get_rsme(parameters):

    processor = ModelProcessor(parameters)
    rsme, mspe = processor.get_rsme(parameters)

    result = dict(status='OK', error_text='', rsme=rsme, mspe=mspe, description='model rsme is recieved')

    return result


def get_model_parameters(parameters):

    processor = ModelProcessor(parameters)
    model_parameters = processor.get_model_parameters(parameters)

    result = dict(status='OK', error_text='', model_parameters=model_parameters,
                  description='model parameters is recieved')

    return result


def set_db_connector(parameters):
    global DB_CONNECTOR
    if not DB_CONNECTOR:
        DB_CONNECTOR = db_connector.Connector(parameters, initialize=True)


def get_factor_analysis_data(parameters):

    processor = ModelProcessor(parameters)
    fa, indicator_description, graph_bin = processor.get_factor_analysis_data(parameters)

    get_graph = parameters.get('get_graph')

    result = dict(status='OK', error_text='', result=fa, indicator_description=indicator_description,
                  description='model factor analysis data recieved')

    if get_graph:
        result['graph_data'] = base64.b64encode(graph_bin).decode(encoding='utf-8')

    return result


def disable_gpu():
    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        print(visible_devices)
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except Exception as e:
        error_text = 'Invalid device or cannot modify virtual devices once initialized.'
        raise ProcessorException('{}. {}'.format(error_text, str(e)))