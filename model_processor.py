import settings_controller
from logger import ProcessorException as ProcessorException
import db_connector
import numpy as np

import pandas as pd
import os
import math

from job_processor import JobProcessor

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
import eli5
from eli5.sklearn import PermutationImportance
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Normalization
from tensorflow.keras import optimizers

from tensorflow import keras

import matplotlib.pyplot as plt
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

        need_to_update = bool(parameters.get('need_to_update'))

        self.model = Model(model_description['id'],
                           model_description['name'],
                           model_description['x_indicators'],
                           model_description['y_indicators'],
                           need_to_update=need_to_update)
        retrofit = parameters.get('retrofit')
        date_from = parameters.get('date_from')
        calculate_fi = parameters.get('calculate_fi')

        history = self.model.fit(parameters.get('epochs'),
                                 parameters.get('validation_split'),
                                 retrofit=retrofit,
                                 date_from=date_from,
                                 calculate_fi=calculate_fi)

        return history

    def predict(self, parameters):

        model_description = parameters.get('model')

        if not model_description:
            raise ProcessorException('model is not in parameters')

        self.model = Model(model_description['id'])

        inputs = parameters.get('inputs')
        if not inputs:
            raise ProcessorException('inputs not found in parameters')

        get_graph = parameters.get('get_graph')
        graph_data = parameters.get('graph_data')

        prediction, y_columns, graph_bin = self.model.predict(inputs, get_graph=get_graph, graph_data=graph_data)

        return prediction, y_columns, graph_bin

    def calculate_feature_importances(self, parameters):

        model_description = parameters.get('model')

        if not model_description:
            raise ProcessorException('model is not in parameters')

        self.model = Model(model_description['id'])

        result = self.model.calculate_feature_importances(date_from=parameters.get('date_from'))

        return result

    def get_feature_importances(self, parameters):

        model_description = parameters.get('model')

        if not model_description:
            raise ProcessorException('model is not in parameters')

        self.model = Model(model_description['id'])

        result = self.model.get_feature_importances()

        return result


class Model:

    def __init__(self, model_id, name='', x_indicators=None, y_indicators=None, need_to_update=False):
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
            self.x_indicators = []
            self.y_indicators = []
            self.periods = []
            self.organisations = []
            self.scenarios = []
            self.x_columns = []
            self.y_columns = []

        if x_indicators:
            self.x_indicators = self._data_processor.get_indicator_ids(x_indicators)

        if y_indicators:
            self.y_indicators = self._data_processor.get_indicator_ids(y_indicators)

        self._inner_model = None
        self._epochs = 0

        self.need_to_update = not description_from_db or need_to_update
        self.graph_file_name = 'graph.png'

    def update_model(self, data):
        if self.need_to_update:
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

    def fit(self, epochs=100, validation_split=0.2, retrofit=False, date_from=None, calculate_fi=False):

        if not retrofit:
            date_from = None
        else:
            date_from = datetime.datetime.strptime(date_from, '%d.%m.%Y')

        data = self._data_processor.read_raw_data(self.x_indicators + self.y_indicators, date_from)
        if retrofit and self.need_to_update:
            raise ProcessorException('Model can not be updated when retrofit')

        self._epochs = epochs or 100

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
        if not retrofit:
            self._data_processor.write_columns(self.model_id, x_columns, y_columns)

        # scaler = self._get_scaler(retrofit=retrofit)
        # x = scaler.fit_transform(x)

        inner_model = self._get_inner_model(x.shape[1], y.shape[1], retrofit=retrofit)

        if isinstance(inner_model, keras.models.Sequential):
            normalizer = inner_model.layers[0]
            normalizer.adapt(x)

        inner_model.compile(optimizer=optimizers.Adam(learning_rate=0.1), loss='MeanSquaredError',
                            metrics=['RootMeanSquaredError'])

        validation_split = validation_split or 0.2

        history = inner_model.fit(x, y, epochs=self._epochs, verbose=2, validation_split=validation_split)

        # self._data_processor.write_scaler(self.model_id, scaler)
        self._data_processor.write_inner_model(self.model_id, inner_model)

        if calculate_fi:
            self.calculate_feature_importances(x, y, x_columns, date_from)

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

        x, x_pd, x_columns = self._data_processor.get_x_for_prediction(data, additional_data)

        # scaler = self._get_scaler(True)
        # x_sc = scaler.transform(x)

        inner_model = self._get_inner_model(retrofit=True)

        if isinstance(inner_model, keras.models.Sequential):
            normalizer = inner_model.layers[0]
            normalizer.adapt(x)

        y = inner_model.predict(x)

        data = x_pd.copy()
        data[self.y_columns] = y

        graph_bin = None
        if get_graph:

            x_graph, y_graph = self._get_dataframe_for_graph(data, graph_data['x_indicator'], graph_data['y_indicator'])

            x_label = graph_data['x_indicator']['report_type'] + '\n' + graph_data['x_indicator']['indicator']
            y_label = graph_data['y_indicator']['report_type'] + '\n' + graph_data['y_indicator']['indicator']
            self._make_graph(x_graph, y_graph, x_label, y_label)

            graph_bin = self.read_graph_file()

        outputs = data.drop(self.x_columns, axis=1)

        return outputs.to_dict('records'), self.y_columns, graph_bin

    def calculate_feature_importances(self, x=None, y=None, x_columns=None, date_from=None):

        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            data = self._data_processor.read_raw_data(self.x_indicators + self.y_indicators, date_from)
            additional_data = {'x_indicators': self.x_indicators,
                               'y_indicators': self.y_indicators,
                               'periods': self.periods,
                               'organisations': self.organisations,
                               'scenarios': self.scenarios,
                               'x_columns': self.x_columns,
                               'y_columns': self.y_columns}
            x, y, x_columns, y_columns = self._data_processor.get_x_y_for_fitting(data, additional_data)

        fi_model = KerasRegressor(build_fn=self._get_model_for_feature_importances, nb_epoch=1000, verbose=2)
        fi_model.fit(x, y)

        perm = PermutationImportance(fi_model, random_state=42).fit(x, y)

        fi = pd.DataFrame(perm.feature_importances_, columns=['feature_importance'])
        fi['feature'] = x_columns
        fi = fi.sort_values(by='feature_importance', ascending=False)
        fi['indicator'] = fi['feature'].apply(self._data_processor.get_indicator_name)
        fi['report_type'] = fi['feature'].apply(self._data_processor.get_indicator_report_type)

        fi = fi.to_dict('records')
        self._data_processor.write_feature_importances(self.model_id, fi)

        return fi

    def get_feature_importances(self):
        fi = self._data_processor.read_feature_importances(self.model_id)
        if not fi:
            raise ProcessorException('Feature importances is not calculates')

        return fi

    def _get_scaler(self, retrofit=False):

        if retrofit:
            scaler = self._data_processor.read_scaler(self.model_id)
            if not scaler:
                scaler = MinMaxScaler()
        else:
            scaler = MinMaxScaler()

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
        model_copy = self._create_inner_model(len(self.x_columns), len(self.y_columns))
        model_copy.compile(optimizer=optimizers.Adam(learning_rate=0.1), loss='MeanSquaredError',
                           metrics=['RootMeanSquaredError'])
        return model_copy

    @staticmethod
    def _create_inner_model(inputs_number, outputs_number):

        model = Sequential()
        normalizer = Normalization(axis=-1)
        model.add(normalizer)
        model.add(Dense(150, activation="relu", input_shape=(inputs_number,), name='dense_1'))
        model.add(Dense(outputs_number, activation="linear", name='dense_4'))

        return model

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

        # plt.show()
        fig.savefig(self.graph_file_name)

    def read_graph_file(self):

        f = open(self.graph_file_name, 'rb')
        result = f.read()
        f.close()

        return result

    def _get_dataframe_for_graph(self, data, x_indicator, y_indicator):

        x_indicator_descr = self._data_processor.get_indicator_from_name_type(x_indicator['indicator'],
                                                                              x_indicator['report_type'])
        x_column = x_indicator_descr['indicator_id'] + '_value'

        y_indicator_descr = self._data_processor.get_indicator_from_name_type(y_indicator['indicator'],
                                                                              y_indicator['report_type'])
        y_column = y_indicator_descr['indicator_id'] + '_value'

        data = data[[x_column, y_column]]

        data = data.sort_values(by=[x_column])
        data_sum = data.groupby([x_column]).sum()
        data_sum = data_sum.reset_index()
        data_count = data.groupby([x_column]).count()
        data_count = data_count.reset_index()

        data_sum[y_column] = data_sum[y_column]/data_count[y_column]

        data = data_sum

        return np.array(data[x_column]), np.array(data[y_column])


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

        if not indicators:
            return []

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

    def read_raw_data(self, indicators, date_from):
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

    def get_x_y_for_fitting(self, data, additional_data):

        data = pd.DataFrame(data)

        data_grouped, data_grouped_values = self._prepare_dataset_group(data)

        indicators = additional_data['x_indicators'] + additional_data['y_indicators']

        data = self._prepare_dataset_merge(data_grouped, data_grouped_values, indicators)

        additional_data['years'] = list(set([self._get_year(period) for period in additional_data['periods']]))
        additional_data['months'] = list(set([self._get_month(period) for period in additional_data['periods']]))

        data = self._prepare_dataset_one_hot_encode(data, additional_data)

        inputs = data.copy()
        inputs = inputs.drop([indicator + '_value' for indicator in additional_data['y_indicators']], axis=1)

        outputs = data.copy()
        outputs = outputs[[indicator + '_value' for indicator in additional_data['y_indicators']]]

        x = inputs.drop(['organisation', 'scenario', 'period', 'month', 'year'], axis=1)
        x_columns = list(x.columns)
        x = x.to_numpy()

        y_columns = list(outputs.columns)
        y = outputs.to_numpy()

        return x, y, x_columns, y_columns

    def get_x_for_prediction(self, data, additional_data):

        data['indicator_id'] = data[['indicator', 'report_type']].apply(self._get_indicator_id_one_arg, axis=1)
        data['loading_date'] = None # datetime.datetime.now()
        data_grouped, data_grouped_values = self._prepare_dataset_group(data)

        indicators = additional_data['x_indicators']

        data = self._prepare_dataset_merge(data_grouped, data_grouped_values, indicators)

        additional_data['years'] = list(set([self._get_year(period) for period in additional_data['periods']]))
        additional_data['months'] = list(set([self._get_month(period) for period in additional_data['periods']]))

        data = self._prepare_dataset_one_hot_encode(data, additional_data)

        x = data.drop(['organisation', 'scenario', 'period', 'month', 'year'], axis=1)
        x_columns = list(x.columns)
        x = x.to_numpy()

        return x, data, x_columns

    def write_columns(self, model_id, x_columns, y_columns):
        self._db_connector.write_model_columns(model_id, x_columns, y_columns)

    def write_scaler(self, model_id, scaler):
        scaler_packed = pickle.dumps(scaler, protocol=pickle.HIGHEST_PROTOCOL)
        self._db_connector.write_model_scaler(model_id, scaler_packed)

    def write_feature_importances(self, model_id, feature_importances):
        self._db_connector.write_model_fi(model_id, feature_importances)

    def read_feature_importances(self, model_id):
        model_description = self.read_model_description_from_db(model_id)
        return model_description.get('feature_importances')

    def read_scaler(self, model_id):
        model_description = self.read_model_description_from_db(model_id)
        scaler_packed = model_description['scaler']
        return pickle.loads(scaler_packed)

    def write_inner_model(self, model_id, inner_model):
        if not os.path.isdir('tmp'):
            os.mkdir('tmp')

        inner_model.save('tmp/model')

        zipf = zipfile.ZipFile('tmp/model.zip', 'w', zipfile.ZIP_DEFLATED)
        self._zipdir('tmp/model', zipf)
        zipf.close()

        with open('tmp/model.zip', 'rb') as f:
            model_packed = f.read()

        self._db_connector.write_inner_model(model_id, model_packed)

        os.remove('tmp/model.zip')
        shutil.rmtree('tmp/model')

    def read_inner_model(self, model_id):

        if not os.path.isdir('tmp'):
            os.mkdir('tmp')

        model_description = self.read_model_description_from_db(model_id)
        inner_model = model_description['inner_model']
        with open('tmp/model.zip', 'wb') as f:
            f.write(inner_model)

        with zipfile.ZipFile('tmp/model.zip', 'r') as zip_h:
            zip_h.extractall('tmp/model')

        inner_model = keras.models.load_model('tmp/model')

        return inner_model

    @staticmethod
    def _prepare_dataset_group(dataset):

        columns_to_drop = ['indicator', 'report_type']
        if '_id' in list(dataset.columns):
            columns_to_drop.append('_id')

        dataset.drop(columns_to_drop, axis=1, inplace=True)
        dataset.rename({'indicator_id': 'indicator'}, axis=1, inplace=True)

        data_grouped_values = dataset.groupby(['indicator', 'organisation', 'scenario', 'period'])
        data_grouped_values = data_grouped_values.max()
        data_grouped_values = data_grouped_values.reset_index()

        data_grouped = dataset.groupby(['organisation', 'scenario', 'period']).max()
        data_grouped = data_grouped.reset_index()
        data_grouped = data_grouped[['organisation', 'scenario', 'period']]

        return data_grouped, data_grouped_values

    def _prepare_dataset_merge(self, dataset, dataset_grouped_values, indicators):

        for ind in indicators:
            data_grouped_ind = dataset_grouped_values[(dataset_grouped_values['indicator'] == ind)]
            dataset = dataset.merge(data_grouped_ind, how='left', on=['organisation', 'period', 'scenario'])
            dataset = dataset.rename({'value': '{}_value'.format(ind)}, axis=1)
            dataset = dataset.drop(['version', 'indicator', 'loading_date'], axis=1)

        dataset = dataset.fillna(0)
        dataset = self._add_month_year_to_data(dataset)

        return dataset

    def _prepare_dataset_one_hot_encode(self, dataset, additional_data):

        fields_dict = {'organisation': additional_data['organisations'],
                       'year': additional_data['years'],
                       'month': additional_data['months']}

        for field_name, field_values in fields_dict.items():
            dataset = self._one_hot_encode(dataset, field_name, field_values)

        return dataset

    def _add_month_year_to_data(self, dataset):
        dataset['month'] = dataset['period'].apply(self._get_month)
        dataset['year'] = dataset['period'].apply(self._get_year)

        return dataset

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
    prediction, y_columns, graph_bin = processor.predict(parameters)

    result = dict(status='OK', error_text='', result=prediction, description='model predicted')

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
    fi = processor.get_feature_importances(parameters)

    result = dict(status='OK', error_text='', result=fi, description='model feature importances recieved')

    return result


def set_db_connector(parameters):
    global DB_CONNECTOR
    if not DB_CONNECTOR:
        DB_CONNECTOR = db_connector.Connector(parameters, initialize=True)
