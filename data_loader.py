import datetime
import pandas as pd
import hashlib

import db_connector
from logger import ProcessorException
from job_processor import JobProcessor
import settings_controller

DB_CONNECTORS = []


class LoadingProcessor:

    def __init__(self, loading_id, parameters):

        self.loading_id = loading_id

        print(self.loading_id)

        self._db_connector = get_db_connector(parameters)

        self._loading_parameters = self._get_loading_parameters_from_db()
        self._current_package = None

        self._new_loading = not self._loading_parameters
        if not self._new_loading:
            self._packages = self._get_packages_from_db()

        self._do_not_check = parameters.get('do_not_check')
        if not self._do_not_check:
            self._check_loading()

    def initialize_loading(self, loading, packages):

        if not self._new_loading:
            raise ProcessorException('Loading is already initialized!')

        self._initialize_loading_in_db(loading, packages)
        self._new_loading = False

        # if self.type == 'full':
        #     self._drop_raw_data()

        result_info = {'loading': self._loading_parameters, 'packages': self._packages}

        return self._replace_date_to_str_in_info(result_info)

    def load_package_data(self, package):

        if self._new_loading:
            raise ProcessorException('Loading is not initialized!')

        self._initialize_current_package(package)

        overwrite = self._loading_parameters['type'] == 'full'
        append = not self._loading_parameters['status'] == 'registered' and overwrite

        self._set_package_status('in_process')

        raw_data = self._current_package['data']

        if not raw_data:
            raise ProcessorException('Package data is not found')

        try:

            pd_data, indicators, analytics, analytic_keys = self._preprocess_raw_data(raw_data)

            for indicator_el in indicators:
                self._db_connector.write_indicator(indicator_el)

            for analytic_el in analytics:
                self._db_connector.write_analytic(analytic_el)

            for analytic_key in analytic_keys:
                self._db_connector.write_analytic_key(analytic_key)

            raw_data = pd_data.to_dict('records')
            self._db_connector.write_raw_data(raw_data, overwrite=overwrite, append=append)

            self._set_package_status('loaded')

        except Exception as ex:

            self._set_package_status('error', str(ex))
            raise ex

        current_package = self._current_package.copy()
        current_package.pop('data')
        result_info = {'loading': self._loading_parameters, 'packages': self._packages,
                       'current_package': current_package}
        return self._replace_date_to_str_in_info(result_info)

    def get_loading_info(self):

        if self._new_loading:
            raise ProcessorException('Loading is not initialized!')

        result_info = {'loading': self._loading_parameters, 'packages': self._packages}
        return self._replace_date_to_str_in_info(result_info)

    def set_loading_parameters(self, loading):

        if self._new_loading:
            raise ProcessorException('Loading is not initialized!')

        if loading.get('start_date'):
            loading['start_date'] = datetime.datetime.strptime(loading['start_date'], '%d.%m.%Y %H:%M:%S')
        if loading.get('end_date'):
            loading['end_date'] = datetime.datetime.strptime(loading['end_date'], '%d.%m.%Y %H:%M:%S')

        current_loading = self._db_connector.read_loading(loading['id'])
        if current_loading:
            current_loading.update(loading)
        else:
            current_loading = loading

        self._loading_parameters = current_loading
        self._check_loading()

        self._db_connector.write_loading(current_loading)

    def set_package_parameters(self, package):

        if self._new_loading:
            raise ProcessorException('Loading is not initialized!')

        if package.get('start_date'):
            package['start_date'] = datetime.datetime.strptime(package['start_date'], '%d.%m.%Y %H:%M:%S')
        if package.get('end_date'):
            package['end_date'] = datetime.datetime.strptime(package['end_date'], '%d.%m.%Y %H:%M:%S')

        current_index = -1
        for ind in range(len(self._packages)):
            if self._packages[ind]['id'] == package['id']:
                current_index = ind
                break

        if package.get('remove'):
            if current_index != - 1:
                self._packages.pop(current_index)

            self._check_loading()
            self._db_connector.delete_package(package)
        else:
            if current_index != - 1:
                current_package = self._packages[current_index]
                current_package.update(package)
            else:
                current_package = package
                self._packages.append(current_package)

            self._check_loading()
            self._db_connector.write_package(current_package)

    def delete_loading_parameters(self):

        if self._new_loading:
            raise ProcessorException('Loading is not initialized!')

        self._db_connector.delete_loading(self.loading_id)

        self.loading_id = ''
        self._loading_parameters = None
        self._current_package = None
        self._packages = None

    def _get_loading_parameters_from_db(self):
        return self._db_connector.read_loading(self.loading_id)

    def _get_packages_from_db(self):
        return self._db_connector.read_loading_packages(self.loading_id)

    def _initialize_loading_in_db(self, loading, packages):

        c_loading = loading.copy()

        if c_loading['start_date']:
            c_loading['start_date'] = self._str_to_date(c_loading['start_date'])
        if c_loading['end_date']:
            c_loading['end_date'] = self._str_to_date(c_loading['end_date'])

        self._loading_parameters = c_loading

        fields = ['id', 'status', 'number', 'quantity_of_objects',
                  'start_date', 'end_date']
        self.type = loading['type']

        c_packages = []

        for par_package in packages:
            package = {field: par_package[field] for field in fields}
            package['loading_id'] = self.loading_id
            if package['start_date']:
                package['start_date'] = self._str_to_date(package['start_date'])
            if package['end_date']:
                package['end_date'] = self._str_to_date(package['end_date'])
            c_packages.append(package)

        self._packages = c_packages

        self._db_connector.write_loading(self._loading_parameters)
        self._db_connector.write_packages(self._packages)

    def _initialize_current_package(self, package):

        fields = ['id', 'status', 'number', 'quantity_of_objects',
                  'start_date', 'end_date', 'data']

        self._current_package = {field: package[field] for field in fields}

        if self._current_package['start_date']:
            self._current_package['start_date'] = self._str_to_date(self._current_package['start_date'])
        if self._current_package['end_date']:
            self._current_package['end_date'] = self._str_to_date(self._current_package['end_date'])

    def _check_loading(self):

        if not self._do_not_check:
            return

        if self._new_loading:
            if self._loading_parameters['status'] != 'registered':
                raise ProcessorException('Status must be '
                                         '"registered" for new loading. '
                                         'But now status is "{}"'.format(self._loading_parameters['status']))

            errors = list(filter(lambda x: x['status'] != 'registered', self._packages))

            if errors:
                raise ProcessorException('All packages must be in status "registered"')
        else:
            if self._loading_parameters['status'] == 'registered':
                errors = list(filter(lambda x: x['status'] != 'registered', self._packages))

                if errors:
                    raise ProcessorException('All packages must be in status "registered"')

    def _preprocess_raw_data(self, raw_data):

        pd_data = pd.DataFrame(raw_data)

        pd_data = self.add_short_ids_to_raw_data(pd_data)

        pd_data['indicator'] = pd_data[['indicator', 'indicator_short_id']].apply(
            self.add_short_id_to_indicator, axis=1)

        pd_data['ind'] = pd_data.index
        pd_data_grouped = pd_data[['indicator_short_id', 'ind']].groupby(['indicator_short_id'], axis=0, as_index=False).min()
        pd_data_grouped = pd_data_grouped.merge(pd_data[['ind', 'indicator']], on='ind', how='inner')

        indicators = list(pd_data_grouped['indicator'])

        pd_data['analytics'] = pd_data['analytics'].apply(self._add_short_id_to_analytics)

        pd_data_grouped = pd_data[['analytics_key_id', 'ind']].groupby(['analytics_key_id'], axis=0,
                                                                       as_index=False).min()
        pd_data_grouped = pd_data_grouped.loc[pd_data_grouped['analytics_key_id'] != '']
        pd_data_grouped = pd_data_grouped.merge(pd_data[['ind', 'analytics']], on='ind', how='inner')

        pd_data_grouped = pd_data_grouped.rename({'analytics_key_id': 'short_id'}, axis=1)
        analytic_keys = pd_data_grouped[['short_id', 'analytics']].to_dict('records')

        analytics = []
        analytic_ids = []
        for analytic_dict in analytic_keys:
            for an_el in analytic_dict['analytics']:
                if an_el['short_id'] not in analytic_ids:
                    analytic_ids.append(an_el['short_id'])
                    el_copy = an_el.copy()
                    el_copy.pop('kind')
                    analytics.append(el_copy)

        pd_data = pd_data.drop('ind', axis=1)

        return pd_data, indicators, analytics, analytic_keys

    def _drop_raw_data(self):
        self._db_connector.drop_collection('raw_data')

    def add_short_ids_to_raw_data(self, raw_data):

        raw_data['indicator_short_id'] = raw_data['indicator'].apply(self._make_short_id_from_dict)
        raw_data['analytics'] = raw_data['analytics'].apply(self._add_short_id_to_analytics)

        raw_data['analytics_key_id'] = raw_data['analytics'].apply(self._make_short_id_from_list)

        return raw_data

    @staticmethod
    def add_short_id_to_indicator(ind_value):
        result = ind_value[0]
        result['short_id'] = ind_value[1]
        return result

    def _add_short_id_to_analytics(self, analytics_list):
        for an_el in analytics_list:
            an_el['short_id'] = self._make_short_id_from_dict(an_el)
        return analytics_list

    @staticmethod
    def get_hash(value):
        if not value.replace(' ', ''):
            return ''
        data_hash = hashlib.md5(value.encode())
        return data_hash.hexdigest()[-7:]

    def _make_short_id_from_dict(self, dict_value):
        str_val = dict_value['id'] + dict_value.get('type') or ''
        return self.get_hash(str_val)

    def _make_short_id_from_list(self, list_value):
        if list_value:
            short_id_list = [el['short_id'] for el in list_value]
            short_id_list.sort()
            str_val = ''.join(short_id_list)
            return self.get_hash(str_val)
        else:
            return ''

    def _set_package_status(self, status, error_text=''):

        current_date = datetime.datetime.now()

        self._current_package['status'] = status
        self._current_package['error_text'] = error_text

        if status == 'in_process':
            self._current_package['start_date'] = current_date
            self._current_package['end_date'] = ''
        elif status == 'loaded':
            self._current_package['end_date'] = current_date
        elif status == 'registered':
            self._current_package['start_date'] = ''
        elif status == 'error':
            self._current_package['end_date'] = current_date

        for package in self._packages:
            if package['id'] == self._current_package['id']:
                package.update(self._current_package)
                package.pop('data')

        package = self._current_package.copy()
        package['loading_id'] = self.loading_id
        package.pop('data')

        self._db_connector.write_package(package)

        if status == 'registered':

            for package in self._packages:
                package['status'] = status
                package['loading_id'] = self.loading_id
                package['start_date'] = ''
                package['end_date'] = ''

            self._db_connector.write_packages(self._packages)

            self._loading_parameters['status'] = status
            self._loading_parameters['start_date'] = ''
            self._loading_parameters['end_date'] = ''

            self._db_connector.write_loading(self._loading_parameters)

        elif status == 'in_process':

            if self._loading_parameters['status'] != status:
                self._loading_parameters['status'] = status
                self._loading_parameters['start_date'] = current_date
                self._loading_parameters['end_date'] = ''
                self._db_connector.write_loading(self._loading_parameters)

        elif status == 'loaded':

            not_loaded_packages = list(filter(lambda x: x['status'] != 'loaded', self._packages))
            if not not_loaded_packages:
                self._loading_parameters['status'] = status
                self._loading_parameters['end_date'] = current_date
                self._db_connector.write_loading(self._loading_parameters)

        elif status == 'error':

            self._loading_parameters['status'] = status
            self._loading_parameters['error_text'] = error_text
            self._db_connector.write_loading(self._loading_parameters)

    def _replace_date_to_str_in_info(self, loading_info):
        loading_info['loading']['start_date'] = self._date_to_str(loading_info['loading']['start_date'])
        loading_info['loading']['end_date'] = self._date_to_str(loading_info['loading']['end_date'])

        if loading_info.get('current_package'):
            loading_info['current_package']['start_date'] = self._date_to_str(loading_info['current_package']['start_date'])
            loading_info['current_package']['end_date'] = self._date_to_str(loading_info['current_package']['end_date'])

        for package in loading_info['packages']:
            package['start_date'] = self._date_to_str(package['start_date'])
            package['end_date'] = self._date_to_str(package['end_date'])

        return loading_info

    @staticmethod
    def _date_to_str(date):
        result = ''
        if date:
            result = date.strftime('%d.%m.%Y %H:%M:%S')
        return result

    @staticmethod
    def _str_to_date(date_str):
        return datetime.datetime.strptime(date_str, '%d.%m.%Y %H:%M:%S')


def initialize_loading(parameters):

    if not parameters.get('loading'):
        raise ProcessorException('parameter "loading" is not found in parameters')

    if not parameters.get('packages'):
        raise ProcessorException('parameter "packages" is not found in parameters')

    loading = LoadingProcessor(parameters['loading']['id'], parameters)
    loading_info = loading.initialize_loading(parameters['loading'], parameters['packages'])

    return {'status': 'OK', 'error_text': '', 'description': 'loading initialized', 'result_info': loading_info}


@JobProcessor.job_processing
def load_package(parameters):

    if not parameters.get('package'):
        raise ProcessorException('parameter "package" is not found in parameters')

    loading = LoadingProcessor(parameters['package']['loading_id'], parameters)
    loading_info = loading.load_package_data(parameters['package'])

    return {'status': 'OK', 'error_text': '', 'description': 'package loaded', 'result_info': loading_info}


def get_loading_info(parameters):

    if not parameters.get('loading_id'):
        raise ProcessorException('parameter "loading_id" is not found in parameters')

    loading = LoadingProcessor(parameters['loading_id'], parameters)
    loading_info = loading.get_loading_info()

    return {'status': 'OK', 'error_text': '', 'description': 'loading info get',
            'result_info': loading_info}


def set_loading_parameters(parameters):

    if not parameters.get('loading_id'):
        raise ProcessorException('parameter "loading_id" is not found in parameters')

    if not parameters.get('loading'):
        raise ProcessorException('parameter "loading" is not found in parameters')

    loading = LoadingProcessor(parameters['loading_id'], parameters)
    loading.set_loading_parameters(parameters['loading'])

    return {'status': 'OK', 'error_text': '', 'description': 'loading parameters set'}


def set_package_parameters(parameters):

    if not parameters.get('loading_id'):
        raise ProcessorException('parameter "loading_id" is not found in parameters')

    if not parameters.get('package'):
        raise ProcessorException('parameter "package" is not found in parameters')

    loading = LoadingProcessor(parameters['loading_id'], parameters)
    loading.set_package_parameters(parameters['package'])

    return {'status': 'OK', 'error_text': '', 'description': 'loading parameters set'}


def delete_loading_parameters(parameters):

    if not parameters.get('loading_id'):
        raise ProcessorException('parameter "loading_id" is not found in parameters')

    loading = LoadingProcessor(parameters['loading_id'], parameters)
    loading.delete_loading_parameters()

    return {'status': 'OK', 'error_text': '', 'description': 'loading parameters removed'}


def get_db_connector(parameters=None):

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
