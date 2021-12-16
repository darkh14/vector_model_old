import pymongo
from logger import ProcessorException as ProcessorException
import settings_controller
import numpy as np


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

    def write_inputs_outputs(self, inputs, outputs, model_name='', rewrite=False):

        self._write_data(inputs, 'inputs', rewrite)
        self._write_data(outputs, 'outputs', rewrite)

    def write_model_pd_data(self, inputs, outputs, model_name='', rewrite=False):
        self._write_data(inputs.to_dict('records'), 'pd_inputs', rewrite)
        self._write_data(outputs.to_dict('records'), 'pd_outputs', rewrite)

    def write_x_y(self, X, y, model_name='', rewrite=False):

        self._write_data(self._numpy_to_list_of_dicts(X), 'X', rewrite)
        self._write_data(self._numpy_to_list_of_dicts(y), 'y', rewrite)

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

    def read_x_y(self, model_name=''):

        X = self._read_data('X')
        y = self._read_data('y')

        X = np.array([list(line.values())[1:] for line in X])
        y = np.array([list(line.values())[1:] for line in y])

        return X, y

    def read_model_data(self, model_name=''):

        return self._read_data('inputs'), self._read_data('outputs')

    def write_additional_model_data(self, data, model_name='', rewrite=False):
        for data_name, data_list in data.items():
            self._write_data([{'value': data_element}for data_element in data_list], data_name, rewrite, ['value'])

    def read_additional_data(self, names, model_name=''):
        values = [[item['value'] for item in self._read_data(name)] for name in names]
        return dict(zip(names, values))

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

    def _write_data(self, data, collection_name, rewrite=False, selections=None):

        collection = self.get_collection(collection_name)

        if rewrite:
            collection.delete_many({})
            collection.insert_many(data)
        else:
            for line in data:
                self._write_line(collection_name, line, selections)

    def _read_data(self, collection_name):

        collection = self.get_collection(collection_name)
        return list(collection.find())

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

    @staticmethod
    def _numpy_to_list_of_dicts(np_array):
        if not len(np_array.shape) == 2:
            raise ProcessorException('np array must be 2D array')

        row_indexes = list(map(str, list(range(np_array.shape[1]))))

        return [dict(zip(row_indexes, list(line))) for line in np_array]