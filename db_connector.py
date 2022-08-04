import pymongo
from logger import ProcessorException as ProcessorException
import settings_controller
import hashlib

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

        # print('Database {}, id:{}'.format(self.db_name, self.db_id))

        self.error = ''

        if initialize:
            self.initialize()

    def write_raw_data(self, raw_data, overwrite=False, append=False):
        print('---Loading raw data started---')
        collection = self.get_collection('raw_data')

        for line in raw_data:
            line['loading_date'] = datetime.now()

        if overwrite or append:
            if not append:
                print('---Removing old data---')
                collection.drop()
                print('---Removing old data - finished --')
            print('---Inserting new data---')
            collection.insert_many(raw_data)
            print('---Inserting new data - finished--')
        else:
            print('---Updating data--')
            selections = ['period', 'scenario', 'organisation', 'indicator_short_id', 'analytics_key_id']
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

    def read_indicator_from_type_id(self, indicator_type, indicator_id):
        result = self._read_line('indicators', {'type': indicator_type, 'id': indicator_id})
        if result:
            result.pop('_id')

        return result

    def read_indicator_from_short_id(self, indicator_id):
        result = self._read_line('indicators', {'short_id': indicator_id})
        if result:
            result.pop('_id')

        return result

    def read_analytics_from_short_id(self, analytics_id):
        result = self._read_line('analytics', {'short_id': analytics_id})
        if result:
            result.pop('_id')

        return result

    def read_analytics_from_key_id(self, key_id):
        line = self._read_line('analytic_keys', {'short_id': key_id})

        result = None
        if line:
            result = line.get('analytics')

        return result

    def read_indicator_from_name_type(self, indicator, report_type):
        result = self._read_line('indicators', {'indicator': indicator, 'report_type': report_type})
        if result:
            result.pop('_id')

        return result

    def write_indicator(self, indicator):
        self._write_line('indicators', indicator, selections=['id'])

    def write_analytic(self, analytic):
        self._write_line('analytics', analytic, selections=['id', 'type'])

    def write_analytic_key(self, analytic_key):
        self._write_line('analytic_keys', analytic_key, selections=['short_id'])

    def read_model_description(self, model_id):
        return self._read_line('models', {'model_id': model_id})

    def write_model_description(self, model_description):
        db_model_description = self.read_model_description(model_description['model_id'])
        if db_model_description:
            db_model_description.update(model_description)
            model_description = db_model_description
        self._write_line('models', model_description, ['model_id'])

    def write_model_columns(self, model_id, x_columns, y_columns):
        model_description = self.read_model_description(model_id)
        model_description['x_columns'] = x_columns
        model_description['y_columns'] = y_columns

        self.write_model_description(model_description)

    def write_model_analytics(self, model_id, x_analytics, y_analytics, x_analytic_keys, y_analytic_keys):
        model_description = self.read_model_description(model_id)
        model_description['x_analytics'] = x_analytics
        model_description['y_analytics'] = y_analytics
        model_description['x_analytic_keys'] = x_analytic_keys
        model_description['y_analytic_keys'] = y_analytic_keys

        self.write_model_description(model_description)

    def write_model_fi(self, model_id, fi, fi_grouped):
        model_description = self.read_model_description(model_id)
        model_description['feature_importances'] = {'expanded': fi, 'grouped': fi_grouped}

        self.write_model_description(model_description)

    def write_model_scaler(self, model_id, scaler, is_out=False):
        scaler_name = 'y_scaler' if is_out else 'x_scaler'
        model_description = self.read_model_description(model_id)
        model_description[scaler_name] = scaler

        self.write_model_description(model_description)

    def write_inner_model(self, model_id, inner_model):
        model_description = self.read_model_description(model_id)
        model_description['inner_model'] = inner_model

        self.write_model_description(model_description)

    def write_model_field(self, model_id, field_name, value):
        model_description = self.read_model_description(model_id)
        model_description[field_name] = value

        self.write_model_description(model_description)

    def write_loading(self, loading):
        self._write_line('loadings', loading, ['id'])

    def write_package(self, package):
        self._write_line('packages', package, ['loading_id', 'id'])

    def delete_package(self, package):
        lines_filter = {'loading_id': package['loading_id'], 'id': package['id']}
        self._delete_lines('packages', lines_filter)

    def write_packages(self, packages):

        for package in packages:
            self._write_line('packages', package, ['loading_id', 'id'])

    def read_package(self, loading_id, package_id):
        collection = self.get_collection('packages')

        packages = list(collection.find({'loading_id': loading_id, 'id': package_id}))

        package = None
        if packages is not None and len(packages):
            package = packages[0]
            package.pop('_id')

        return package

    def read_loading_packages(self, loading_id):
        collection = self.get_collection('packages')

        packages = list(collection.find({'loading_id': loading_id}))

        if packages is not None and len(packages):
            for package in packages:
                package.pop('_id')

        return packages

    def read_loading(self, loading_id):
        collection = self.get_collection('loadings')

        loadings = list(collection.find({'id': loading_id}))

        loading = None
        if loadings is not None and len(loadings):
            loading = loadings[0]
            loading.pop('_id')

        return loading

    def delete_loading(self, loading_id):
        self._delete_lines('loadings', {'id': loading_id})
        self._delete_lines('packages', {'loading_id': loading_id})

    def read_raw_data(self, indicators, date_from, ad_filter=None):
        collection = self.get_collection('raw_data')
        db_filter = dict()
        if indicators:
            db_filter['indicator_short_id'] = {'$in': indicators}
        if date_from:
            db_filter['loading_date'] = {'$gte': datetime.strptime(date_from, '%d.%m.%Y')}
        if ad_filter:
            for key, value in ad_filter.items():
                db_filter[key] = {'$in': value}

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

    def delete_model(self, model_id):
        self._delete_lines('models', {'model_id': model_id})

    def initialize(self):
        self._connect()
        self._set_db()
        self._set_collections()

    def get_collection(self, collection_name):

        if not self._initialized:
            raise ProcessorException('connector is not initialized')

        collection = self._collections.get(collection_name)

        if collection is None:
            collection = self._db.get_collection(collection_name)
            self._collections[collection_name] = collection

        if collection is None:
            raise ProcessorException('collection {} not found in db {}'.format(collection_name, self.db_name))

        return collection

    @staticmethod
    def get_short_id(data_str):
        if not data_str.replace(' ', ''):
            return ''
        data_hash = hashlib.md5(data_str.encode())
        return data_hash.hexdigest()[-7:]

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

    def read_data(self, collection_name, data_filter=None):

        collection = self.get_collection(collection_name)

        result = collection.find(data_filter)

        return list(result)

    def _get_collection_names(self):

        if not self._db_is_set:
            raise ProcessorException('db must be selected to get collection names')

        return self._db.list_collection_names()

    def _delete_lines(self, collection_name, del_filter):

        collection = self.get_collection(collection_name)
        if not del_filter:
            del_filter = dict()
        collection.delete_many(del_filter)

    def drop_collection(self, collection_name):
        collection = self.get_collection(collection_name)
        collection.drop()

    def get_collection_quantity(self, collection_name, collection_filter=None):
        collection = self.get_collection(collection_name)

        if not collection_filter:
            collection_filter = dict()

        result = collection.count_documents(collection_filter)

        return result

    @staticmethod
    def _numpy_to_list_of_dicts(np_array):
        if not len(np_array.shape) == 2:
            raise ProcessorException('np array must be 2D array')

        row_indexes = list(map(str, list(range(np_array.shape[1]))))

        return [dict(zip(row_indexes, list(line))) for line in np_array]