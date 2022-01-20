import pymongo
from logger import ProcessorException as ProcessorException
import settings_controller
import numpy as np

from datetime import datetime


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

    def write_raw_data(self, raw_data, model_name='', overwrite=False):
        print('---Loading raw data started---')
        collection = self.get_collection('raw_data')

        for line in raw_data:
            line['loading_date'] = datetime.now()

        if overwrite:
            print('---Removing old data---')
            collection.drop()
            print('---Removing old data - finished --')
            print('---Inserting new data---')
            collection.insert_many(raw_data)
            print('---Inserting new data - finished--')
        else:
            print('---Updating data--')
            selections = ['version', 'period', 'scenario', 'organisation', 'report_type', 'indicator']
            line_num = 1
            for line in raw_data:
                line_filter = {selection: line[selection] for selection in selections}
                collection.update_one(line_filter, {'$set': line}, upsert=True)
                if line_num % 100 == 0:
                    print('\r------Updated {} lines'.format(line_num), end='')
                line_num +=1
            print()
            print('---Updating data - finished--')

        print('---Loading raw data started---')

    def read_indicator_from_id(self, indicator_id):
        result = self._read_line('indicators', {'indicator_id': indicator_id})
        if result:
            result.pop('_id')

        return result

    def read_indicator_from_name_type(self, indicator, report_type):
        result = self._read_line('indicators', {'indicator': indicator, 'report_type': report_type})
        if result:
            result.pop('_id')

        return result

    def write_indicator(self, indicator_id, indicator, report_type):
        line = {'indicator_id': indicator_id, 'indicator': indicator, 'report_type': report_type}
        self._write_line('indicators', line, selections=['indicator_id'])

    def read_model_description(self, model_id):
        return self._read_line('models', {'model_id': model_id})

    def write_model_description(self, model_description):
        self._write_line('models', model_description, ['model_id'])

    def write_model_columns(self, model_id, x_columns, y_columns):
        model_description = self.read_model_description(model_id)
        model_description['x_columns'] = x_columns
        model_description['y_columns'] = y_columns

        self.write_model_description(model_description)

    def write_model_fi(self, model_id, fi):
        model_description = self.read_model_description(model_id)
        model_description['feature_importances'] = fi

        self.write_model_description(model_description)

    def write_model_scaler(self, model_id, scaler):
        model_description = self.read_model_description(model_id)
        model_description['scaler'] = scaler

        self.write_model_description(model_description)

    def write_inner_model(self, model_id, inner_model):
        model_description = self.read_model_description(model_id)
        model_description['inner_model'] = inner_model

        self.write_model_description(model_description)

    def read_raw_data(self, indicators, date_from):
        collection = self.get_collection('raw_data')
        db_filter = {'indicator_id': {'$in': indicators}}
        if date_from:
            db_filter['loading_date'] = {'$gte': datetime.strptime(date_from, '%d.%m.%Y')}

        return list(collection.find(db_filter))

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

    def delete_jobs(self, del_filter):
        self._delete_lines('background_jobs', del_filter)

    def initialize(self):
        self._connect()
        self._set_db()
        self._set_collections()

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

    def _write_data(self, data, collection_name, rewrite=False, selections=None):

        collection = self.get_collection(collection_name)

        if rewrite:
            collection.drop()
            collection.insert_many(data)
        else:
            for line in data:
                self._write_line(collection_name, line, selections)

    def _read_line(self, collection_name, line_filter):

        collection = self.get_collection(collection_name)
        return collection.find_one(line_filter)

    def _read_data(self, collection_name):

        collection = self.get_collection(collection_name)
        return list(collection.find())

    def _get_collection_names(self):

        if not self._db_is_set:
            raise ProcessorException('db must be selected to get collection names')

        return self._db.list_collection_names()

    def _delete_lines(self, collection_name, del_filter):

        collection = self.get_collection(collection_name)
        if not del_filter:
            del_filter = dict()
        collection.delete_many(del_filter)

    @staticmethod
    def _numpy_to_list_of_dicts(np_array):
        if not len(np_array.shape) == 2:
            raise ProcessorException('np array must be 2D array')

        row_indexes = list(map(str, list(range(np_array.shape[1]))))

        return [dict(zip(row_indexes, list(line))) for line in np_array]