# import http_procession
import json
import warnings
import processor

warnings.simplefilter(action='ignore', category=FutureWarning)


def t_application(request_type, start_response):

    parameters_string = make_parameters_string_from_json(request_type)

    environ = dict()
    environ['REQUEST_METHOD'] = 'POST'

    environ['wsgi.input'] = [parameters_string]

    output = processor.process(environ, start_response)

    return output


def make_parameters_string_from_list(parameters):

    return '&'.join(['{}={}'.format(key, str(value)) for key, value in parameters])


def make_parameters_string_from_json(request_type):

    file_path = 'parameters_json/' + request_type + '_parameters.json'

    p_file = open(file_path, 'r', encoding="utf-8")
    parameters_string = p_file.read()
    return parameters_string


def t_start_response(status, headers):
    pass


if __name__ == '__main__':

    request_dict = dict()
    request_dict[1] = 'data_load_package'
    request_dict[2] = 'model_fit'
    request_dict[3] = 'model_predict'
    request_dict[4] = 'job_get_state'
    request_dict[5] = 'job_delete'
    request_dict[6] = 'model_calculate_feature_importances'
    request_dict[7] = 'model_get_feature_importances'
    request_dict[8] = 'model_get_rsme'
    request_dict[9] = 'model_get_factor_analysis_data'
    request_dict[10] = 'data_get_loading_info'
    request_dict[11] = 'data_set_loading_parameters'
    request_dict[12] = 'data_set_package_parameters'
    request_dict[13] = 'data_delete_loading_parameters'
    request_dict[14] = 'data_initialize_loading'
    request_dict[15] = 'model_get_parameters'
    request_dict[16] = 'model_initialize'
    request_dict[17] = 'model_update'
    request_dict[18] = 'model_drop'
    request_dict[19] = 'data_get_raw_data_quantity'
    request_dict[20] = 'data_remove_all_raw_data'
    request_dict[21] = 'model_drop_fitting'

    print('Input number to choose action:')
    for key, value in request_dict.items():
        print(str(key) + ' - "' + value + '"')

    print('')
    request_number = int(input('>> '))

    if request_number != 0:

        request_type = request_dict.get(request_number)

        assert request_type, 'request type not found'

        output = t_application(request_type, t_start_response)

        output_str = output[0].decode()
        output_dict = json.loads(output_str)

        print(output_dict)


