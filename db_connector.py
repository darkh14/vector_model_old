import pymongo
from logger import ProcessorException as ProcessorException
import settings_controller
import numpy as np

import datetime


class Connector:

    def __init__(self, parameters, initialize=True):

        self.db_name = parameters.get('db')
        self.db_id = parameters.get('db_id')
        if not self.db_name and not self.db_id:
            raise ProcessorException('parameter "db" or parameter "db_id" must be set')

        self.uri = parameters.get('uri')
        self._connection = None
        self._is_connected = False
        self._db = None
        self._db_is_set = False
        self._collections = None
        self._collections_is_set = False
        self._initialized = False

        self._settings_controller = settings_controller.Controller()
        if not self.uri:
            self.uri = self._settings_controller.get_parameter('mongo_uri')

        if not self.db_id:
            self.db_id = self._settings_controller.get_db_id(self.db_name)
        self.error = ''

        if initialize:
            self.initialize()

    def write_raw_data(self, raw_data, overwrite=False):

        date_now = datetime.datetime.now()

        collection = self.get_collection('raw_data')

        if overwrite:

            for line in raw_data:
                line['loading_date'] = date_now

            collection.drop()
            collection.insert_many(raw_data)
        else:
            for line in raw_data:
                selections = ['version', 'organisation', 'scenario', 'period', 'report_type', 'indicator']

                line_filter = {selection: line[selection] for selection in selections}

                db_line = collection.find_one(line_filter)
                if not db_line or (db_line and line['value'] != db_line['value']):
                    line['loading_date'] = date_now
                    collection.replace_one(line_filter, line, upsert=True)

    def write_job(self, job_line):
        self._write_line('background_jobs', job_line, ['job_id'])

    def read_jobs(self, job_filter, limit=0):

        lines = []

        collection = self.get_collection('background_jobs')

        finder = collection.find(job_filter)

        if limit:
            finder = finder.limit(limit)

        for line in finder:
            line.pop('_id')
            lines.append(line)

        return lines

    def read_job(self, job_id):
        job_filter = {'job_id': job_id}
        lines = self.read_jobs(job_filter, limit=1)
        return lines[0] if lines else None

    def read_model_data(self, model_name=''):

        return self._read_data('inputs'), self._read_data('outputs')

    def read_model_description(self, model_id):
        return self._read_line('models', {'model_id': model_id})

    def write_model_to_db(self, model_description):
        self._write_line('models', model_description, ['model_id'])

    def write_scaler_to_db(self, scaler, model_id):

        model_description = self.read_model_description(model_id)
        model_description['scaler'] = scaler
        self._write_line('models', model_description, ['model_id'])

    def write_inner_model_to_db(self, inner_model, model_id):

        model_description = self.read_model_description(model_id)
        model_description['inner_model'] = inner_model
        self._write_line('models', model_description, ['model_id'])

    def read_indicator_by_type_and_name(self, report_type, indicator):
        return self._read_line('indicators', {'report_type': report_type, 'indicator': indicator})

    def read_indicator_by_id(self, indicator_id):
        result = self._read_line('indicators', {'indicator_id': indicator_id})
        result.pop('_id')
        return result

    def read_dataset_with_indicators_filter(self, indicators):
        collection = self.get_collection('raw_data')

        dataset = collection.find({'indicator_id': {'$in': indicators}})
        dataset = list(dataset)
        return dataset

    def write_indicator(self, line):
        self._write_line('indicators', line, ['indicator_id'])

    def write_additional_model_data(self, data, model_name='', rewrite=False):
        for data_name, data_list in data.items():
            self._write_data([{'value': data_element}for data_element in data_list], data_name, rewrite, ['value'])

    def read_additional_data(self, names, model_name=''):
        values = [[item['value'] for item in self._read_data(name)] for name in names]
        return dict(zip(names, values))

    def write_model_columns(self, model_id, x_columns, y_columns):
        model_description = self.read_model_description(model_id)
        model_description['x_columns'] = x_columns
        model_description['y_columns'] = y_columns
        self._write_line('models', model_description, ['model_id'])

    def initialize(self):
        self._connect()
        self._set_db()
        self._set_collections()

    def _connect(self, **kwargs):

        if not self.uri:
            raise ProcessorException('an not connect to db, uri is not defined')

        reconnect = kwargs.get('reconnect')

        if reconnect or not self._is_connected:
            self._connection = pymongo.MongoClient(self.uri)
            self._is_connected = True
            self._set_db()

        return True

    def _set_db(self, **kwargs):

        if not self._is_connected:
            raise ProcessorException('connector is not connect to db')

        reset = kwargs.get('reset')

        if reset or not self._db_is_set:
            self._db = self._connection[self.db_id]
            self._db_is_set = True
            return self._db
        else:
            return self._db

    def _set_collections(self, **kwargs):

        if not self._is_connected:
            raise ProcessorException('connector is not connect to db')

        if not self._db_is_set:
            raise ProcessorException('db must be selected to get collection names')

        reset = kwargs.get('reset')

        if reset or not self._collections_is_set:
            self._collections = {collection_name: None for collection_name in self._get_collection_names()}
            self._collections_is_set = True
            self._initialized = True

        return self._collections

    def _write_line(self, collection_name,  line, selections=None):

        collection = self.get_collection(collection_name)

        if not selections:
            collection.insert_one(line)
        else:
            line_filter = {selection: line[selection] for selection in selections}
            collection.replace_one(line_filter, line, upsert=True)

    def _write_data(self, data, collection_name, overwrite=False, selections=None):

        collection = self.get_collection(collection_name)

        if overwrite:
            collection.delete_many({})
            collection.insert_many(data)
        else:
            for line in data:
                self._write_line(collection_name, line, selections)

    def _read_data(self, collection_name):

        collection = self.get_collection(collection_name)
        return list(collection.find())

    def _read_line(self, collection_name, collection_filter):

        collection = self.get_collection(collection_name)
        return collection.find_one(collection_filter)

    def get_collection(self, collection_name):

        if not self._initialized:
            raise ProcessorException('connector is not initialized')

        collection = self._collections.get(collection_name)

        if not collection:
            collection = self._db.get_collection(collection_name)
            self._collections[collection_name] = collection

        if not collection:
            raise ProcessorException('collection {} not found in db {}'.format(collection_name, self.db_name))

        return collection

    def _get_collection_names(self):

        if not self._db_is_set:
            raise ProcessorException('db must be selected to get collection names')

        return self._db.list_collection_names()

    def delete_jobs(self, del_filter):
        self._delete_lines('background_jobs', del_filter)

    def _delete_lines(self, collection_name, del_filter):

        collection = self.get_collection(collection_name)
        if not del_filter:
            del_filter = dict()
        collection.delete_many(del_filter)