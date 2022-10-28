import time

import settings_controller
from logger import ProcessorException as ProcessorException
import db_connector

import numpy as np
import pandas as pd
import os
import psutil
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
# from keras.wrappers.scikit_learn import KerasRegressor
import eli5

from eli5.sklearn import PermutationImportance
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go

import base64

import pickle
import zipfile
import shutil
import datetime

DB_CONNECTORS = []


class ModelProcessor:

    def __init__(self, parameters):

        self._db_connector = get_db_connector(parameters)
        self._data_processor = DataProcessor(self._db_connector)

        self.model = None

    def initialize_model(self, parameters):
        model_description = parameters.get('model')

        if not model_description:
            raise ProcessorException('model is not in parameters')

        self.model = self._get_model(model_description)

        self.model.initialize_model()

    def update_model(self, parameters):
        model_description = parameters.get('model')

        if not model_description:
            raise ProcessorException('model is not in parameters')

        self.model = self._get_model(model_description)

        drop_started_fitting = parameters.get('drop_started_fitting')
        drop_undefined_parameters = parameters.get('drop_undefined_parameters')

        self.model.update_model(model_description, drop_started_fitting=drop_started_fitting,
                                drop_undefined_parameters=drop_undefined_parameters)

    def fit(self, parameters):
        model_description = parameters.get('model')

        if not model_description:
            raise ProcessorException('model is not in parameters')

        self.model = self._get_model(model_description)

        retrofit = parameters.get('retrofit')
        date_from = parameters.get('date_from')

        job_id = parameters.get('job_id') or ''

        history = self.model.fit(epochs=parameters.get('epochs'),
                                 validation_split=parameters.get('validation_split'),
                                 retrofit=retrofit,
                                 date_from=date_from, job_id=job_id)

        return history

    def drop_fitting(self, parameters):
        model_description = parameters.get('model')

        if not model_description:
            raise ProcessorException('model is not in parameters')

        self.model = self._get_model(model_description)

        self.model.drop_fitting(parameters.get('sleep_before'))

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

        job_id = parameters.get('job_id') or ''

        result = self.model.calculate_feature_importances(date_from=parameters.get('date_from'),
                                                          epochs=parameters.get('epochs'), job_id=job_id)

        return result

    def drop_fi_calculation(self, parameters):
        model_description = parameters.get('model')

        if not model_description:
            raise ProcessorException('model is not in parameters')

        self.model = self._get_model(model_description)

        self.model.drop_fi_calculation(parameters.get('sleep_before'))

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

        output_indicator = parameters.get('output_indicator')

        if not output_indicator:
            raise ProcessorException('output indicator is not in parameters')

        input_indicators = parameters.get('input_indicators')

        if not input_indicators:
            raise ProcessorException('input indicators are not in parameters')

        outputs = parameters.get('outputs')

        if not outputs:
            raise ProcessorException('outputs are not in parameters')

        result, graph_data = self.model.get_factor_analysis_data(inputs,
                                                                 input_indicators,
                                                                 outputs,
                                                                 output_indicator,
                                                                 get_graph=parameters.get('get_graph'))

        return result, graph_data

    def drop_model(self, parameters):

        model_description = parameters.get('model')

        if not model_description:
            raise ProcessorException('model is not in parameters')

        self.model = self._get_model(model_description)

        self.model.drop_model()

    def _get_model(self, model_description):

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

        model = model_class(model_description['id'], self._db_connector, model_description)

        return model


class BaseModel:

    __metaclass__ = ABCMeta
    type = ''

    def __init__(self, model_id, database_connector, model_parameters):

        name = model_parameters.get('name') or ''

        x_indicators = model_parameters.get('x_indicators')
        y_indicators = model_parameters.get('y_indicators')

        model_filter = model_parameters.get('filter')

        self.model_id = model_id
        self.db_id = ''

        self.name = name
        self.filter = []

        self.organisations = []
        self.scenarios = []
        self.periods = []

        self.x_indicators = []
        self.y_indicators = []

        self.x_columns = []
        self.y_columns = []

        self.x_analytics = []
        self.y_analytics = []

        self.x_analytic_keys = []
        self.y_analytic_keys = []

        self._db_connector = database_connector

        print('Database {}, id:{}'.format(self._db_connector.db_name, self._db_connector.db_id))

        self._data_processor = DataProcessor(self._db_connector)

        self.initialized = False
        self.is_fit = False

        self.fitting_date = None
        self.fitting_is_started = False
        self.fitting_is_error = False
        self.fitting_start_date = None
        self.fitting_job_id = ''
        self.fitting_job_pid = 0

        self.feature_importances = None
        self.feature_importances_is_calculated = False
        self.fi_calculation_is_started = False
        self.fi_calculation_is_error = False
        self.fi_calculation_job_id = ''
        self.fi_calculation_job_pid = 0

        self.rsme = 0
        self.mspe = 0

        self._field_to_update = ['name', 'type', 'initialized', 'is_fit', 'fitting_is_started', 'fitting_is_error',
                                 'fitting_start_date',
                                 'fitting_date', 'rsme', 'mspe',
                                 'feature_importances_is_calculated', 'fi_calculation_is_started',
                                 'fi_calculation_is_error', 'fi_calculation_job_id', 'fi_calculation_job_pid',
                                 'fitting_is_started',
                                 'filter', 'x_indicators', 'y_indicators', 'periods', 'organisations',
                                 'scenarios', 'x_columns', 'y_columns', 'x_analytics', 'y_analytics',
                                 'x_analytic_keys', 'y_analytic_keys', 'feature_importances',
                                 'fitting_job_id',
                                 'fitting_job_pid']

        description_from_db = self._data_processor.read_model_description_from_db(self.model_id)

        if description_from_db:
            for field in self._field_to_update:
                setattr(self, field, description_from_db.get(field))

        if x_indicators:
            self.x_indicators = self._data_processor.get_indicators_data_from_parameters(x_indicators)

        if y_indicators:
            self.y_indicators = self._data_processor.get_indicators_data_from_parameters(y_indicators)

        self.type = model_parameters['type']

        self._inner_model = None
        self._retrofit = False

        if model_filter:
            self.filter = model_filter

        self.graph_file_name = 'graph.png'
        self.graph_fi_file_name = 'fi_graph.png'
        self.graph_fa_file_name = 'fa_graph.png'

    def initialize_model(self):

        if self.initialized:
            raise ProcessorException('Model is already initialized')

        self.initialized = True
        self.rsme = 0
        self.mspe = 0

        self.feature_importances = None

        self._write_model_to_db()

    def update_model(self, model_parameters=None, drop_started_fitting=False, drop_undefined_parameters=False):

        if not self.initialized:
            raise ProcessorException('Model is not initialized')

        if self.fitting_is_started and not drop_started_fitting:
            raise ProcessorException('Fitting is started, model can not be updated. If you ')

        if drop_undefined_parameters:
            model_description = self._get_default_model_values()
            for key, value in model_description.items():
                setattr(self, key, value)

        if model_parameters:
            x_indicators = model_parameters.get('x_indicators')
            y_indicators = model_parameters.get('y_indicators')

            if x_indicators:
                self.x_indicators = self._data_processor.get_indicators_data_from_parameters(x_indicators)
                model_parameters['x_indicators'] = self.x_indicators

            if y_indicators:
                self.y_indicators = self._data_processor.get_indicators_data_from_parameters(y_indicators)
                model_parameters['y_indicators'] = self.y_indicators

            for key, value in model_parameters.items():
                if key in self._field_to_update:
                    setattr(self, key, value)

        model_description = {field: getattr(self, field) for field in self._field_to_update}

        self._data_processor.write_model_to_db(self.model_id, model_description)

    def update_model_while_fitting(self, data):
        organisations, scenarios, periods = self._data_processor.get_additional_data(data)
        self.periods = periods
        self.organisations = organisations
        self.scenarios = scenarios

        model_description = {field: getattr(self, field) for field in ['organisations', 'periods', 'scenarios']}

        self._data_processor.write_model_to_db(self.model_id, model_description)

    def fit(self, epochs=100, validation_split=0.2, retrofit=False, date_from=None, job_id=''):

        try:

            if not self.initialized:
                raise ProcessorException('Model is not initialized')

            if self.fitting_is_started:
                raise ProcessorException('Fitting is always started')

            print('job_id:  {}'.format(job_id))

            job_id = job_id or ''

            model_description = {'is_fit': False,
                                 'fitting_is_started': True,
                                 'fitting_is_error': False,
                                 'fitting_date': None,
                                 'fitting_start_date': datetime.datetime.now(),
                                 'fitting_job_id': job_id,
                                 'feature_importances_is_calculated': False,
                                 'fi_calculation_is_started': False,
                                 'fi_calculation_is_error': False,
                                 'fi_calculation_job_id': '',
                                 'fi_calculation_job_pid': 0}

            self._set_model_fields_and_write_to_db(model_description)

            current_pid = os.getpid()
            self.fitting_job_pid = current_pid
            self._data_processor.write_model_field(self.model_id, 'fitting_job_pid', self.fitting_job_pid)

            self.fit_model(epochs=epochs, validation_split=validation_split, retrofit=retrofit, date_from=date_from)

            model_description = {'is_fit': True,
                                 'fitting_is_started': False,
                                 'fitting_is_error': False,
                                 'fitting_date': datetime.datetime.now(),
                                 'fitting_start_date': None,
                                 'fitting_job_pid': 0}

            self._set_model_fields_and_write_to_db(model_description)

        except Exception as ex:

            model_description = {'is_fit': False,
                                 'fitting_is_started': False,
                                 'fitting_is_error': True,
                                 'fitting_date': None,
                                 'fitting_start_date': None,
                                 'feature_importances_is_calculated': False,
                                 'fi_calculation_is_started': False,
                                 'fi_calculation_is_error': False,
                                 'fi_calculation_job_id': '',
                                 'fi_calculation_job_pid': 0}

            self._set_model_fields_and_write_to_db(model_description)

            raise ex

    def drop_fitting(self, sleep_before=0):

        if sleep_before:
            time.sleep(sleep_before)

        if not self.initialized:
            raise ProcessorException('Model is not initialized')

        if not self.fitting_is_started and not self.is_fit:
            raise ProcessorException('Wrong model status. Model must be fitted or fitting must be started for dropping')

        if self.fitting_is_started and self.fitting_job_pid:

            try:
                process = psutil.Process(self.fitting_job_pid)
                process.terminate()
            except psutil.NoSuchProcess:
                print('No such process pid {}'.format(self.fitting_job_pid))
            except psutil.Error as error:
                print('Process termination error. {}'.format(str(error)))

            if self.fitting_job_id:
                job_line = self._data_processor.get_job(self.fitting_job_id)
                job_line['status'] = 'interrupted'
                self._data_processor.set_job(job_line)

        model_description = {'is_fit': False,
                             'fitting_is_started': False,
                             'fitting_is_error': False,
                             'fitting_date': None,
                             'fitting_start_date': datetime.datetime.now(),
                             'fitting_job_id': '',
                             'fitting_job_pid': 0,
                             'feature_importances_is_calculated': False,
                             'fi_calculation_is_started': False,
                             'fi_calculation_is_error': False,
                             'fi_calculation_job_id': '',
                             'fi_calculation_job_pid': 0}

        self._set_model_fields_and_write_to_db(model_description)

    def drop_fi_calculation(self, sleep_before=0):

        if sleep_before:
            time.sleep(sleep_before)

        if not self.initialized:
            raise ProcessorException('Model is not initialized')

        if not self.is_fit:
            raise ProcessorException('Model is not fit')

        if not self.fi_calculation_is_started and not self.feature_importances_is_calculated:
            raise ProcessorException('Wrong model status. Model feature importances must be calculated'
                                     ' or feature importances calculation must be started for dropping')

        if self.fi_calculation_is_started and self.fi_calculation_job_pid:

            try:
                process = psutil.Process(self.fi_calculation_job_pid)
                process.terminate()
            except psutil.NoSuchProcess:
                print('No such process pid {}'.format(self.fi_calculation_job_pid))
            except psutil.Error as error:
                print('Process termination error. {}'.format(str(error)))

            if self.fi_calculation_job_id:
                job_line = self._data_processor.get_job(self.fi_calculation_job_id)
                job_line['status'] = 'interrupted'
                self._data_processor.set_job(job_line)

        model_description = {'feature_importances_is_calculated': False,
                             'fi_calculation_is_started': False,
                             'fi_calculation_is_error': False,
                             'fi_calculation_job_id': '',
                             'fi_calculation_job_pid': 0}

        self._set_model_fields_and_write_to_db(model_description)

    @abstractmethod
    def fit_model(self, epochs=100, validation_split=0.2, retrofit=False, date_from=None):
        """method for fitting model"""

    @abstractmethod
    def predict(self, inputs, get_graph=False, graph_data=None, additional_parameters=None):
        """method for predicting data from model"""

    def calculate_feature_importances(self, date_from=None, epochs=1000, retrofit=False, validation_split=0.2, job_id=''):

        result = {}
        try:
            if not self.initialized:
                raise ProcessorException('Error of calculating feature importances. Model is not initialized')

            if not self.is_fit:
                raise ProcessorException('Error of calculating feature importances. Model is not fit. '
                                         'Train the model before calculating')
            job_id = job_id or ''
            model_description = {'feature_importances_is_calculated': False,
                                 'fi_calculation_is_started': True,
                                 'fi_calculation_is_error': False,
                                 'fi_calculation_job_id': job_id,
                                 'fi_calculation_job_pid': 0}

            self._set_model_fields_and_write_to_db(model_description)

            current_pid = os.getpid()
            self.fi_calculation_job_pid = current_pid
            self._data_processor.write_model_field(self.model_id, 'fi_calculation_job_pid', self.fi_calculation_job_pid)

            result = self.calculate_fi_after_check(date_from=date_from, epochs=epochs, retrofit=retrofit,
                                                   validation_split=validation_split)

            model_description = {'feature_importances_is_calculated': True,
                                 'fi_calculation_is_started': False,
                                 'fi_calculation_is_error': False,
                                 'fi_calculation_job_id': '',
                                 'fi_calculation_job_pid': 0}

            self._set_model_fields_and_write_to_db(model_description)

        except Exception as ex:

            model_description = {'feature_importances_is_calculated': False,
                                 'fi_calculation_is_started': False,
                                 'fi_calculation_is_error': True,
                                 'fitting_job_id': '',
                                 'fitting_job_pid': 0}

            self._set_model_fields_and_write_to_db(model_description)

            raise ex

        return result

    @abstractmethod
    def calculate_fi_after_check(self, date_from=None, epochs=1000, retrofit=False, validation_split=0.2):
        """method for calculating feature importances after checking"""

    def _calculate_fi_from_model(self, fi_model, x, y, x_columns):
        perm = PermutationImportance(fi_model, random_state=42).fit(x, y)

        fi = pd.DataFrame(perm.feature_importances_, columns=['feature_importance'])
        fi['feature'] = x_columns
        fi = fi.sort_values(by='feature_importance', ascending=False)
        fi['indicator'] = fi['feature'].apply(self._data_processor.get_indicator_name)

        fi = fi.to_dict('records')
        self._data_processor.write_feature_importances(self.model_id, fi)

        return fi

    def get_feature_importances(self, get_graph=False):

        if not self.feature_importances_is_calculated:
            raise ProcessorException('Feature importances is not calculated')

        fi = self._data_processor.read_feature_importances(self.model_id)

        graph_bin = None
        # if get_graph:
        #     graph_bin = self._get_fi_graph_bin(fi)

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

        initialized = self.initialized

        if initialized:
            rsme = self._data_processor.read_model_field(self.model_id, 'rsme')
            mspe = self._data_processor.read_model_field(self.model_id, 'mspe')

            feature_importances = self._data_processor.read_model_field(self.model_id, 'feature_importances')

            is_fit = self._data_processor.read_model_field(self.model_id, 'is_fit')
            fitting_is_started = self._data_processor.read_model_field(self.model_id, 'fitting_is_started')
            fitting_is_error = self._data_processor.read_model_field(self.model_id, 'fitting_is_error')
            fitting_date = self._data_processor.read_model_field(self.model_id, 'fitting_date')

            if fitting_date:
                fitting_date = fitting_date.strftime('%d.%m.%Y %H:%M:%S')

            fitting_start_date = self._data_processor.read_model_field(self.model_id, 'fitting_start_date')
            if fitting_start_date:
                fitting_start_date = fitting_start_date.strftime('%d.%m.%Y %H:%M:%S')

            fitting_job_id = self._data_processor.read_model_field(self.model_id, 'fitting_job_id')

            feature_importances_is_calculated = bool(self._data_processor.read_model_field(self.model_id,
                                                                                   'feature_importances_is_calculated'))

            fi_calculation_is_started = bool(self._data_processor.read_model_field(self.model_id,
                                                                                   'fi_calculation_is_started'))

            fi_calculation_is_error = bool(self._data_processor.read_model_field(self.model_id,
                                                                                 'fi_calculation_is_error'))
            fi_calculation_job_id = bool(self._data_processor.read_model_field(self.model_id, 'fi_calculation_job_id'))

        else:
            rsme = 0
            mspe = 0

            feature_importances = None

            is_fit = False
            fitting_is_started = False
            fitting_is_error = False
            fitting_date = None
            fitting_start_date = None
            fitting_job_id = ''

            feature_importances_is_calculated = False
            fi_calculation_is_started = False

            fi_calculation_is_error = False
            fi_calculation_job_id = ''



        model_parameters = {'initialized': initialized, 'rsme': rsme, 'mspe': mspe, 'is_fit': is_fit,
                            'fitting_date': fitting_date,
                            'fitting_is_started': fitting_is_started,
                            'fitting_is_error': fitting_is_error,
                            'fitting_start_date': fitting_start_date,
                            'feature_importances': feature_importances,
                            'feature_importances_is_calculated': feature_importances_is_calculated,
                            'fi_calculation_is_started': fi_calculation_is_started,
                            'fi_calculation_is_error': fi_calculation_is_error,
                            'fi_calculation_job_id': fi_calculation_job_id,
                            'fitting_job_id': fitting_job_id}

        return model_parameters

    def drop_model(self):

        if not self.initialized:
            raise ProcessorException('Model is not initialized')

        self._data_processor.delete_model(self.model_id)

    def get_factor_analysis_data(self, inputs, input_indicators, outputs, output_indicator_id, get_graph=False):

        if not self.initialized:
            raise ProcessorException('Error of calculating factor analysis data. Model is not initialized')

        if not self.is_fit:
            raise ProcessorException('Error of calculating factor analysis data. Model is not fit. '
                                     'Train the model before calculating')

        return self.get_factor_analysis_data_from_model(inputs, input_indicators, outputs, output_indicator_id,
                                                        get_graph)

    @abstractmethod
    def get_factor_analysis_data_from_model(self, inputs, input_indicators, outputs, output_indicator_id,
                                            get_graph=False):
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

    def _get_data_for_fa_graph(self, result_data, outputs):

        result_data = pd.DataFrame(result_data)
        result_data['title'] = result_data['indicator'].apply(lambda x: x['name'])
        result_data['order'] = list(range(2, result_data.shape[0]+2))

        result_data.drop(['indicator'], axis=1, inplace=True)

        base_line = {'title': 'Базовый', 'value': outputs['based']['value'], 'order': 1}
        calculated_line = {'title': 'Расчетный', 'value': outputs['calculated']['value'], 'order': result_data.shape[0]+2}

        result_data = pd.concat([result_data, pd.DataFrame([base_line, calculated_line])])

        result_data = result_data.sort_values('order')

        return result_data

    def _get_indicator_name_from_description(self, indicator_id):
        descr_lines = list(filter(lambda x: x['id'] == indicator_id, self.x_indicators + self.y_indicators))
        return descr_lines[0]['name']

    def _get_fa_graph_bin(self, values, out_indicator_name):

        x_list = list(values['title'])
        y_list = list(values['value'])

        text_list = []
        for index, item in enumerate(y_list):
            if item > 0 and index != 0 and index != len(y_list) - 1:
                text_list.append('+{0:.2f}'.format(y_list[index]))
            else:
                text_list.append('{0:.2f}'.format(y_list[index]))

        for index, item in enumerate(text_list):
            if item[0] == '+' and index != 0 and index != len(text_list) - 1:
                text_list[index] = '<span style="color:#2ca02c">' + text_list[index] + '</span>'
            elif item[0] == '-' and index != 0 and index != len(text_list) - 1:
                text_list[index] = '<span style="color:#d62728">' + text_list[index] + '</span>'
            if index == 0 or index == len(text_list) - 1:
                text_list[index] = '<b>' + text_list[index] + '</b>'

        dict_list = []
        for i in range(0, 1200, 200):
            dict_list.append(dict(
                type="line",
                line=dict(
                    color="#666666",
                    dash="dot"
                ),
                x0=-0.5,
                y0=i,
                x1=6,
                y1=i,
                line_width=1,
                layer="below"))

        fig = go.Figure(go.Waterfall(
            name="Factor analysis", orientation="v",
            measure=["absolute", *(values.shape[0]-2) * ["relative"], "total"],
            x=x_list,
            y=y_list,
            text=text_list,
            textposition="outside",
            connector={"line": {"color": 'rgba(0,0,0,0)'}},
            increasing={"marker": {"color": "#2ca02c"}},
            decreasing={"marker": {"color": "#d62728"}},
            totals={'marker': {"color": "#9467bd"}},
            textfont={"family": "Open Sans, light",
                      "color": "black"
                      }
        ))

        f = fig.update_layout(
            title=
            {'text': '<b>Факторный анализ</b><br><span style="color:#666666">{}</span>'.format(out_indicator_name)},
            showlegend=False,
            height=650,
            font={
                'family': 'Open Sans, light',
                'color': 'black',
                'size': 14
            },
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis_title="руб.",
            shapes=dict_list
        )

        fig.update_xaxes(tickangle=-45, tickfont=dict(family='Open Sans, light', color='black', size=14))
        fig.update_yaxes(tickangle=0, tickfont=dict(family='Open Sans, light', color='black', size=14))

        graph_str = fig.to_html()

        return graph_str

    def _write_model_to_db(self, model_fields=None):

        if not model_fields:
            model_fields = self._field_to_update

        model_description = {el: getattr(self, el) for el in model_fields}

        self._data_processor.write_model_to_db(self.model_id, model_description)

    def _set_model_fields_and_write_to_db(self, model_description: dict):
        self._set_model_fields(model_description)
        self._write_model_to_db(model_fields=model_description.keys())

    def _set_model_fields(self, model_description: dict):
        for key, value in model_description.items():
            setattr(self, key, value)

    @staticmethod
    def _get_default_model_values() -> dict:

        values = {}
        values['periods'] = []
        values['organisations'] = []
        values['scenarios'] = []

        values['x_columns'] = []
        values['y_columns'] = []

        values['x_analytics'] = []
        values['y_analytics'] = []

        values['x_analytic_keys'] = []
        values['y_analytic_keys'] = []

        values['is_fit'] = False
        values['fitting_is_started'] = False
        values['fitting_is_error'] = False

        values['feature_importances_is_calculated'] = False
        values['fi_calculation_is_started'] = False

        values['fi_calculation_is_error'] = False
        values['fi_calculation_job_id'] = ''
        values['fi_calculation_job_pid'] = 0

        values['fitting_start_date'] = None
        values['fitting_date'] = None

        values['fitting_job_id'] = ''
        values['fitting_job_pid'] = 0

        return values


class NeuralNetworkModel(BaseModel):

    type = 'neural_network'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._epochs = 0
        self._validation_split = 0.2

        self._temp_input = None

    def fit_model(self, epochs=100, validation_split=0.2, retrofit=False, date_from=None):

        x, y = self._prepare_for_fit(retrofit, date_from)

        if not x.any():
            raise ProcessorException('There is no data for fitting')

        if not y.any():
            raise ProcessorException('There is no labels for fitting')

        self._inner_model = self._get_inner_model(x.shape[1], y.shape[1])
        self._epochs = epochs or 1000
        self._validation_split = validation_split or 0.2

        history = self._inner_model.fit(x, y, epochs=self._epochs, verbose=2, validation_split=self._validation_split)

        self._write_after_fit(x, y)

        return history.history

    @staticmethod
    def _compile_model(model):
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='MeanSquaredError',
                      metrics=['RootMeanSquaredError'])

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

        inner_model = self._get_inner_model()

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
                       'y_analytics': self.y_analytics,
                       'x_analytic_keys': self.x_analytic_keys,
                       'y_analytic_keys': self.y_analytic_keys}

        return outputs.to_dict('records'), description, graph_bin

    def calculate_fi_after_check(self, date_from=None, epochs=1000, retrofit=False, validation_split=0.2):

        if not retrofit:
            date_from = None
        else:
            date_from = datetime.datetime.strptime(date_from, '%d.%m.%Y')

        indicator_filter = [ind_data['short_id'] for ind_data in self.x_indicators + self.y_indicators]

        db_filter = {key: value for key, value in self.filter.items() if key not in ['date_from', 'date_to']}

        data = self._data_processor.read_raw_data(indicator_filter, date_from=date_from, ad_filter=db_filter)
        additional_data = {'x_indicators': self.x_indicators,
                           'y_indicators': self.y_indicators,
                           'periods': self.periods,
                           'organisations': self.organisations,
                           'scenarios': self.scenarios,
                           'x_columns': self.x_columns,
                           'y_columns': self.y_columns,
                           'filter': self.filter}
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

        db_filter = {key: value for key, value in self.filter.items() if key not in ['date_from', 'date_to']}

        data = self._data_processor.read_raw_data(indicator_filter, date_from=date_from, ad_filter=db_filter)

        if not data:
            raise ProcessorException('There are no data for fitting. Check indicators, analytics and other '
                                     'parameters of fitting. Also check loading data')
        if not retrofit:
            self.update_model_while_fitting(data)

        additional_data = {'model_id': self.model_id,
                           'x_indicators': self.x_indicators,
                           'y_indicators': self.y_indicators,
                           'periods': self.periods,
                           'organisations': self.organisations,
                           'scenarios': self.scenarios,
                           'x_columns': self.x_columns,
                           'y_columns': self.y_columns,
                           'filter': self.filter}
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

        fi = pd.DataFrame(perm.feature_importances_, columns=['error_delta'])
        fi['feature'] = x_columns
        fi = fi.sort_values(by='error_delta', ascending=False)

        fi = fi.loc[fi['feature'] != 'month'].copy()

        fi[['indicator_short_id', 'indicator']] = fi[['feature']].apply(self._data_processor.get_indicator_data_from_fi,
                                                                        axis=1, result_type='expand')
        fi[['analytic_key_id', 'analytics']] = fi[['feature']].apply(self._data_processor.get_analytics_data_from_fi,
                                                                     axis=1, result_type='expand')

        fi['influence_factor'] = fi['error_delta'].apply(self._get_influence_factor_from_error_delta)

        if_sum = fi['influence_factor'].sum()
        fi['influence_factor'] = fi['influence_factor']/if_sum

        fi_ind = fi.copy()
        fi_ind['error_delta_plus'] = fi_ind['error_delta'].apply(lambda x: x if x > 0 else 0)

        fi_ind['count'] = 1

        fi_ind = fi_ind.groupby(['indicator_short_id'], as_index=False).sum()
        fi_ind['indicator'] = fi_ind['indicator_short_id'].apply(self._data_processor.get_indicator_description_from_short_id)

        fi_ind['error_delta'] = fi_ind['error_delta']/fi_ind['count']

        fi_ind = fi_ind.drop(['error_delta_plus', 'count'], axis=1)

        fi = fi.to_dict('records')
        fi_ind = fi_ind.to_dict('records')
        self._data_processor.write_feature_importances(self.model_id, fi, fi_ind)

        return fi

    @staticmethod
    def _calculate_rsme(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def _get_influence_factor_from_error_delta(error_delta):

        if error_delta < 0:
            return 0
        else:
            return math.log(error_delta + 1)  # 1 - math.exp(-error_delta)

    def _get_scaler(self, retrofit=False, is_out=False):

        if retrofit:
            scaler = self._data_processor.read_scaler(self.model_id, is_out)
            if not scaler:
                scaler = StandardScaler()
        else:
            scaler = StandardScaler()

        return scaler

    def _get_inner_model(self, inputs_number=0, outputs_number=0):

        inner_model = self._data_processor.read_inner_model(self.model_id)
        if not inner_model:
            inner_model = self._create_inner_model(inputs_number, outputs_number)

            self._compile_model(inner_model)

        self._inner_model = inner_model

        return inner_model

    def _get_model_for_feature_importances(self):
        # model_copy = clone_model(self._inner_model) #
        model_copy = self._create_inner_model(len(self.x_columns), len(self.y_columns))
        model_copy.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='MeanSquaredError',
                           metrics=['RootMeanSquaredError'])
        return model_copy

    def get_factor_analysis_data_from_model(self, inputs, input_indicators, outputs, output_indicator,
                                            get_graph=False):

        result_data = []
        used_indicator_ids = []

        additional_data = {'x_indicators': self.x_indicators,
                           'y_indicators': self.y_indicators,
                           'periods': self.periods,
                           'organisations': self.organisations,
                           'scenarios': self.scenarios,
                           'x_columns': self.x_columns,
                           'y_columns': self.y_columns}

        output_indicator_short_id = self._data_processor.get_indicator_short_id(output_indicator['type'],
                                                                                output_indicator['id'])

        output_columns = [col for col in self.y_columns if col.split('_')[1] == output_indicator_short_id]

        input_data = pd.DataFrame(inputs)
        main_periods = input_data[['period',
                                   'is_main_period']].loc[input_data['is_main_period'] == True]['period'].unique()

        main_periods = list(main_periods)

        additional_data['main_periods'] = main_periods
        additional_data['output_columns'] = output_columns

        output_base = self._get_output_value_for_fa(input_data, outputs, used_indicator_ids, additional_data)

        for indicator_element in input_indicators:

            ind_short_id = self._data_processor.get_indicator_short_id(indicator_element['type'],
                                                                       indicator_element['id'])

            if not ind_short_id:
                raise ProcessorException('Indicator {}, type {}, id {} is not in model indicators')

            output_value = self._get_output_value_for_fa(input_data, outputs, used_indicator_ids, additional_data,
                                                         ind_short_id)

            result_element = {'indicator': indicator_element, 'value': output_value - output_base}

            result_data.append(result_element)
            used_indicator_ids.append(ind_short_id)

            output_base = output_value

        graph_string = ''
        if get_graph:
            graph_data = self._get_data_for_fa_graph(result_data, outputs)
            graph_string = self._get_fa_graph_bin(graph_data, output_indicator['name'])

        return result_data, graph_string

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

    def _get_output_value_for_fa(self, input_data, outputs, used_indicator_ids, additional_data,
                                 current_ind_short_id=''):

        c_input_data = input_data.copy()

        c_input_data['scenario'] = outputs['calculated']['name']
        c_input_data['periodicity'] = outputs['calculated']['periodicity']

        c_input_data = self._data_processor.add_short_ids_to_raw_data(c_input_data)

        c_input_data['current_indicator_short_id'] = current_ind_short_id
        c_input_data['used_indicator_ids'] = None
        c_input_data['used_indicator_ids'] = c_input_data['used_indicator_ids'].apply(lambda x: used_indicator_ids)

        c_input_data['value'] = c_input_data[['value_base',
                                              'value_calculated',
                                              'used_indicator_ids',
                                              'indicator_short_id',
                                              'current_indicator_short_id']].apply(self._get_value_for_fa, axis=1)

        c_input_data = c_input_data.drop(['value_base', 'value_calculated', 'used_indicator_ids',
                                          'current_indicator_short_id'], axis=1)

        # encode_fields = {'organisation': 'organisations', 'year': 'years', 'month': 'months'}
        encode_fields = None

        x, x_pd = self._data_processor.get_x_for_prediction(c_input_data, additional_data, encode_fields)

        inner_model = self._get_inner_model()

        y = inner_model.predict(x)

        data = x_pd.copy()
        data[self.y_columns] = y

        main_periods = additional_data['main_periods']
        output_columns = additional_data['output_columns']

        output_data = data.loc[data['period'].isin(main_periods)].copy()

        output_data = output_data[output_columns]
        output_data['value'] = output_data.apply(sum, axis=1)

        output_value = output_data['value'].sum()

        return output_value

    @staticmethod
    def _get_value_for_fa(input_parameters):

        (value_based, value_calculated, used_indicator_ids,
         indicator_short_id, current_indicator_short_id) = input_parameters
        value = 0
        if current_indicator_short_id == indicator_short_id:
            value = value_calculated # value_calculated - value_based
        elif indicator_short_id in used_indicator_ids:
            value = value_calculated
        else:
            value = value_based

        return value

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
        self._data_processor = PeriodicDataProcessor(self._db_connector)

        model_parameters = args[2]
        model_description = self._data_processor.read_model_description_from_db(model_parameters['id'])
        if model_description:
            self.past_history = model_description.get('past_history')
            self.future_target = model_description.get('future_target')
        else:
            self.past_history = model_parameters.get('past_history')
            self.future_target = model_parameters.get('future_target')

    def initialize_model(self):
        super().initialize_model()

        model_description = {'past_history': self.past_history, 'future_target': self.future_target}
        self._data_processor.write_model_to_db(self.model_id, model_description)

    def update_model(self, model_parameters=None, drop_started_fitting=False):
        super().update_model(model_parameters, drop_started_fitting)
        field_to_update = ['past_history', 'future_target']
        for key, value in model_parameters.items():
            if key in field_to_update:
                setattr(self, key, value)

        model_description = {field: getattr(self, field) for field in field_to_update}
        self._data_processor.write_model_to_db(self.model_id, model_description)

    def update_model_while_fitting(self, data):
        super().update_model_while_fitting(data)
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

        inner_model = self._get_inner_model(x.shape[-2:], y.shape[-2])

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

        inner_model = self._get_inner_model()

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

    def calculate_fi_after_check(self, date_from=None, epochs=1000, retrofit=False, validation_split=0.2):

        raise ProcessorException('Feature importances is not allowed for periodic neural network')

        return None

    def get_factor_analysis_data_from_model(self, inputs, input_indicators, outputs, output_indicator_id, get_graph=False):

        raise ProcessorException('factor analysis is not allowed for periodic neural network')

        return None

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


class LinearModel(NeuralNetworkModel):

    type = 'linear_regression'

    @staticmethod
    def _create_inner_model(inputs_number, outputs_number):

        model = Sequential()

        model.add(Dense(inputs_number, activation="relu", input_shape=(inputs_number,), name='dense_1'))
        model.add(Dense(outputs_number, activation="linear", name='dense_last'))

        return model

    @staticmethod
    def _compile_model(model):
        model.compile(optimizer=optimizers.Adam(learning_rate=0.01), loss='MeanSquaredError',
                      metrics=['RootMeanSquaredError'])


class IdProcessor(LoadingProcessor):

    def __init__(self):
        pass


class DataProcessor:

    def __init__(self, database_connector):

        self._db_connector = database_connector
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
                raise ProcessorException('indicator {}, id {}, type {}  not found in indicators'.format(parameters_line.get('name'),
                                                                                                        parameters_line['id'],
                                                                                                        parameters_line['type']))
            result_line.update(parameters_line)
            result.append(result_line)

        return result

    def get_indicator_description_from_type_id(self, indicator_type, indicator_id):

        result = self._db_connector.read_indicator_from_type_id(indicator_type, indicator_id)

        return result

    def get_indicator_description_from_short_id(self, short_id):

        result = self._db_connector.read_indicator_from_short_id(short_id)

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

    def get_job(self, job_id):
        return self._db_connector.read_job(job_id)

    def set_job(self, job_line):
        return self._db_connector.write_job(job_line)

    def delete_model(self, model_id):
        self._db_connector.delete_model(model_id)

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

        # TODO: Optimize period filter
        data['period_date'] = data['period'].apply(lambda x: datetime.datetime.strptime(x, '%d.%m.%Y'))
        if additional_data['filter'].get('date_from'):
            filter_date_from = datetime.datetime.strptime(additional_data['filter']['date_from'], '%d.%m.%Y')
            data = data[data['period_date'] >= filter_date_from]
        if additional_data['filter'].get('date_to'):
            filter_date_to = datetime.datetime.strptime(additional_data['filter']['date_to'], '%d.%m.%Y')
            data = data[data['period_date'] <= filter_date_to]

        additional_data['years'] = list(set([self._get_year(period) for period in additional_data['periods']]))
        additional_data['months'] = list(set([self._get_month(period) for period in additional_data['periods']]))

        data.drop('period_date', axis=1, inplace=True)

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

    def write_feature_importances(self, model_id, feature_importances, feature_importances_grouped):
        self._db_connector.write_model_fi(model_id, feature_importances, feature_importances_grouped)

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
            inner_model = model_description.get('inner_model')
            if inner_model:
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

                if with_analytics:
                    data_str_a = data_str.loc[(data_str['analytics'] == an_el)]
                else:
                    data_str_a = data_str

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

    def get_indicator_short_id(self, indicator_type, indicator_id):
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

        x_an_keys_ids = []
        y_an_keys_ids = []

        for column in columns:
            if len(column) > 4:
                if column[4:11] in x_ind_ids:
                    c_col_list = column.split('_')
                    if len(c_col_list) >= 3 and c_col_list[2] == 'an':
                        short_id = c_col_list[3]
                        analytics = self._get_analytics_description_from_key_id(short_id)
                        if short_id not in x_an_keys_ids:
                            x_analytic_keys.append({'short_id': short_id, 'analytics': analytics})
                            x_an_keys_ids.append(short_id)

                if column[4:11] in y_ind_ids:
                    c_col_list = column.split('_')
                    if len(c_col_list) >= 3 and c_col_list[2] == 'an':
                        short_id = c_col_list[3]
                        analytics = self._get_analytics_description_from_key_id(short_id)
                        if short_id not in y_an_keys_ids:
                            y_analytic_keys.append({'short_id': short_id, 'analytics': analytics})
                            y_an_keys_ids.append(short_id)

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


def initialize_model(parameters):
    processor = ModelProcessor(parameters)
    processor.initialize_model(parameters)
    return {'status': 'OK', 'error_text': '', 'description': 'model initialized'}


def update_model(parameters):
    processor = ModelProcessor(parameters)
    processor.update_model(parameters)
    return {'status': 'OK', 'error_text': '', 'description': 'model updated'}


@JobProcessor.job_processing
def fit(parameters):
    processor = ModelProcessor(parameters)
    history = processor.fit(parameters)
    return {'status': 'OK', 'error_text': '', 'description': 'model fitted', 'history': history}


def drop_fitting(parameters):

    processor = ModelProcessor(parameters)
    processor.drop_fitting(parameters)

    result = dict(status='OK', error_text='', description='fitting is dropped')

    return result


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


def drop_fi_calculation(parameters):

    processor = ModelProcessor(parameters)
    processor.drop_fi_calculation(parameters)

    result = dict(status='OK', error_text='', description='feature importances calculation is dropped')

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


def drop_model(parameters):

    processor = ModelProcessor(parameters)
    processor.drop_model(parameters)

    result = dict(status='OK', error_text='', description='model is dropped')

    return result


def get_model_parameters(parameters):

    processor = ModelProcessor(parameters)
    model_parameters = processor.get_model_parameters(parameters)

    result = dict(status='OK', error_text='', model_parameters=model_parameters,
                  description='model parameters is recieved')

    return result


def get_db_connector(parameters):

    global DB_CONNECTORS

    db_name = parameters.get('db')

    if not db_name:
        raise 'Error of getting db connector. "db_name" not in parameters'

    current_settings_controller = settings_controller.Controller()
    db_id = current_settings_controller.get_db_id(db_name)

    result_list = list(filter(lambda x: x.db_id == db_id, DB_CONNECTORS))

    if not result_list:
        result = db_connector.Connector(parameters, initialize=True)
        DB_CONNECTORS.append(result)
    else:
        result = result_list[0]

    return result


def get_factor_analysis_data(parameters):

    processor = ModelProcessor(parameters)
    fa, graph_data = processor.get_factor_analysis_data(parameters)

    result = dict(status='OK', error_text='', result=fa, graph_data=graph_data,
                  description='model factor analysis data recieved')

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