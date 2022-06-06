import datetime
import pandas as pd
import hashlib

import db_connector
from logger import ProcessorException

DB_CONNECTOR = None


class LoadingProcessor:

    def __init__(self, loading_id, parameters):

        self.loading_id = loading_id

        set_db_connector(parameters)
        self._db_connector = DB_CONNECTOR

        self._loading_parameters = self._get_loading_parameters_from_db()

        self._new_loading = False
        if not self._loading_parameters:
            self._initialize_loading_in_db(parameters)
            self._new_loading = True
        else:
            self._packages = self._get_packages_from_db()

        fields = ['id', 'status', 'number', 'start_date', 'end_date', 'quantity_of_objects', 'data']
        self._current_package = {field: parameters['package_' + field] for field in fields}
        if parameters['package_start_date']:
            self._current_package['start_date'] = datetime.datetime.strptime(parameters['package_start_date'],
                                                                             '%d.%m.%Y %H:%M:%S')
        if parameters['package_end_date']:
            self._current_package['end_date'] = datetime.datetime.strptime(parameters['package_end_date'],
                                                                           '%d.%m.%Y %H:%M:%S')
        self._check_loading()

    def _get_loading_parameters_from_db(self):
        return self._db_connector.read_loading(self.loading_id)

    def _get_packages_from_db(self):
        return self._db_connector.read_loading_packages(self.loading_id)

    def _initialize_loading_in_db(self, parameters):

        if not parameters.get('packages'):
            raise ProcessorException('parameter "packages" is not found in parameters')

        fields = ['id', 'type', 'status', 'quantity_of_packages',
                  'start_date', 'end_date']

        loading = {field: parameters['loading_' + field] for field in fields}
        loading['id'] = self.loading_id

        if loading['start_date']:
            loading['start_date'] = datetime.datetime.strptime(loading['start_date'], '%d.%m.%Y %H:%M:%S')
        if loading['end_date']:
            loading['end_date'] = datetime.datetime.strptime(loading['end_date'], '%d.%m.%Y %H:%M:%S')

        self._loading_parameters = loading

        packages = []
        fields = ['id', 'status', 'number', 'quantity_of_objects',
                  'start_date', 'end_date']

        for par_package in parameters['packages']:
            package = {field: par_package[field] for field in fields}
            package['loading_id'] = self.loading_id
            if package['start_date']:
                package['start_date'] = datetime.datetime.strptime(package['start_date'], '%d.%m.%Y %H:%M:%S')
            if package['end_date']:
                package['end_date'] = datetime.datetime.strptime(package['end_date'], '%d.%m.%Y %H:%M:%S')
            packages.append(package)

        self._packages = packages

        self._db_connector.write_loading(loading)
        self._db_connector.write_packages(packages)

    def _check_loading(self):

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

    def load_package_data(self, package_data=None):

        self._set_package_status('in_process')

        overwrite = self._loading_parameters['status'] == 'registered' \
                    and self._loading_parameters['type'] == 'full'
        append = not self._loading_parameters['status'] == 'registered' \
                    and self._loading_parameters['type'] == 'full'

        raw_data = package_data or self._current_package['data']

        if not raw_data:
            raise ProcessorException('Package data is not found')

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

    def _set_package_status(self, status):

        self._current_package['status'] = status
        package = self._current_package.copy()
        package['loading_id'] = self.loading_id
        package.pop('data')
        self._db_connector.write_package(package)

        if status == 'registered':

            for package in self._packages:
                package['status'] = status
                package['loading_id'] = self.loading_id

            self._db_connector.write_packages(self._packages)

            self._loading_parameters['status'] = status
            self._db_connector.write_loading(self._loading_parameters)

        elif status == 'in_process':

            if self._loading_parameters['status'] != status:
                self._loading_parameters['status'] = status
                self._db_connector.write_loading(self._loading_parameters)

        elif status == 'loaded':

            not_loaded_packages = list(filter(lambda x: x['status'] != 'loaded', self._packages))
            if not not_loaded_packages:
                self._loading_parameters['status'] = status
                self._db_connector.write_loading(self._loading_parameters)

        elif status == 'error':

            self._loading_parameters['status'] = status
            self._db_connector.write_loading(self._loading_parameters)

def load_package(parameters):

    if not parameters.get('loading_id'):
        raise ProcessorException('parameter "loading_id" is not found in parameters')

    loading = LoadingProcessor(parameters['loading_id'], parameters)
    loading.load_package_data()

    return {'status': 'OK', 'error_text': '', 'description': 'package loaded'}


def set_db_connector(parameters):
    global DB_CONNECTOR
    if not DB_CONNECTOR:
        DB_CONNECTOR = db_connector.Connector(parameters, initialize=True)