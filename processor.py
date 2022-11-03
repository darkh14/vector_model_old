import json
# import os

import sys
sys.path.append('./VectorModel')
import model_processor
import job_processor
import data_loader
from logger import ProcessorException as ProcessorException
from settings_controller import Controller
import traceback

SERVICE_NAME = 'vector_model'


class Processor:

    def __init__(self):
        self._environ = ''
        self._start_response = None
        self.parameters = {}
        self.output = {}
        self.settings_controller = Controller()

    def process(self, environ, start_response):

        if self._set_parameters_from_environ(environ, start_response):

            self.process_with_parameters()

        return self.transform_output_parameters_to_str(self.output)

    def process_with_parameters(self, parameters=None):

        if parameters:
            self.parameters = parameters

        if self.parameters.get('service_name') != SERVICE_NAME:
            raise ProcessorException('"service name" is not allowed. '
                                     'correct service name is {}'.format(SERVICE_NAME))

        request_type = self.parameters['request_type']

        if not request_type:
            raise ProcessorException('"request type" is not in parameters. parameter "request type" is required')

        method = self._requests_methods().get(request_type)

        if not method:
            raise ProcessorException('method for request type {} is not found'.format(request_type))

        result = method(self.parameters)

        self.output.update(result)

        return self.output

    def _set_parameters_from_environ(self, environ, start_response):

        self._start_response = start_response
        self._environ = environ

        if self._environ.get('REQUEST_METHOD') == 'POST':

            content_length = int(environ.get('CONTENT_LENGTH')) if environ.get('CONTENT_LENGTH') else 0

            par_string = ''

            if content_length:
                par_string = self._environ['wsgi.input'].read(content_length)
            else:
                par_list = self._environ.get('wsgi.input')
                if par_list:
                    for par_element in par_list:
                        par_string = par_element

            if par_string:
                self.parameters = self._parameters_from_json(par_string)

            if not self.parameters.get('db'):
                raise ProcessorException('parameter "db" not found')

        else:
            raise ProcessorException('Request method must be "post"')

        return True

    def _set_parameters(self, parameters):

        self.parameters = parameters

        return True

    def transform_output_parameters_to_str(self, output, start_response=None):

        output_str = json.dumps(output, ensure_ascii=False).encode()
        output_len = len(output_str)

        _start_response = start_response or self._start_response

        _start_response('200 OK', [('Content-type', 'text/html'), ('Content-Length', str(output_len))])

        return [output_str]

    def _requests_methods(self):
        result = dict()

        result['data_load_package'] = data_loader.load_package
        result['model_fit'] = model_processor.fit
        result['model_predict'] = model_processor.predict
        result['model_calculate_feature_importances'] = model_processor.calculate_feature_importances
        result['model_get_feature_importances'] = model_processor.get_feature_importances
        result['model_get_rsme'] = model_processor.get_rsme
        result['model_get_factor_analysis_data'] = model_processor.get_factor_analysis_data
        result['job_get_state'] = job_processor.job_get_state
        result['job_delete'] = job_processor.job_delete
        result['ping'] = self.ping
        result['data_get_loading_info'] = data_loader.get_loading_info
        result['data_set_loading_parameters'] = data_loader.set_loading_parameters
        result['data_set_package_parameters'] = data_loader.set_package_parameters
        result['data_delete_loading_parameters'] = data_loader.delete_loading_parameters
        result['data_initialize_loading'] = data_loader.initialize_loading
        result['data_get_raw_data_quantity'] = data_loader.get_raw_data_quantity
        result['data_remove_all_raw_data'] = data_loader.remove_all_raw_data
        result['model_get_parameters'] = model_processor.get_model_parameters
        result['model_initialize'] = model_processor.initialize_model
        result['model_update'] = model_processor.update_model
        result['model_drop'] = model_processor.drop_model
        result['model_drop_fitting'] = model_processor.drop_fitting
        result['model_drop_fi_calculation'] = model_processor.drop_fi_calculation

        return result

    @staticmethod
    def _parameters_from_json(xml_string):

        x_str = xml_string
        if ord(x_str[0]) == 65279:
            x_str = x_str[1:]

        x_str = x_str.encode('utf-8-sig')
        return json.loads(x_str)

    @staticmethod
    def ping(parameters):
        return {'status': 'OK', 'error_text': '', 'description': 'ping OK'}


def process(environ, start_response=None):

    if not start_response:
        start_response = f_start_response

    processor = Processor()
    # try:
    #     output = processor.process(environ, start_response)
    # except ProcessorException as e:
    #     output = dict()
    #     output['status'] = 'error'
    #     output['error_text'] = str(e)
    #     output = processor.transform_output_parameters_to_str(output, start_response=start_response)
    # except Exception as e:
    #     output = dict()
    #     output['status'] = 'error'
    #     output['error_text'] = traceback.format_exc()
    #     output = processor.transform_output_parameters_to_str(output, start_response=start_response)
    output = processor.process(environ, start_response)
    return output


def f_start_response():
    pass


def process_with_parameters(parameters):

    processor = Processor()
    try:
        output = processor.process_with_parameters(parameters)
    except ProcessorException as e:
        output = dict()
        output['status'] = 'error'
        output['error_text'] = str(e)

    except Exception as e:
        output = dict()
        output['status'] = 'error'
        output['error_text'] = traceback.format_exc()

    print(output)
    return output
