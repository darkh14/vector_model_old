import subprocess
import traceback
import inspect
import sys
import os
import uuid
from db_connector import Connector as DbConnector
from importlib import import_module
from logger import ProcessorException
import time
from datetime import datetime


class JobProcessor:

    def __init__(self, parameters):
        pass

    @classmethod
    def job_processing(cls, func):

        def wrapper(wrapper_parameters, **kwargs):

            if wrapper_parameters.get('background_job'):
                return_value = JobProcessor.execute_method_in_background(func, wrapper_parameters, **kwargs)
            else:
                return_value = func(wrapper_parameters, **kwargs)

            return return_value

        return wrapper

    @classmethod
    def execute_method_in_background(cls, func, parameters, **kwargs):

        new_job_id = str(uuid.uuid4())

        logger = StdOutErrLogger(new_job_id, parameters=parameters)
        logger.clear_old_logs()
        out_file_name, err_file_name = logger.get_log_file_names()

        module_name, function_name = func.__module__, func.__name__

        new_line = {'job_id': new_job_id,
                    'start_date': datetime.now(),
                    'finish_date': None,
                    'job_name': '.'.join([module_name, function_name]),
                    'status': 'created',
                    'result': None,
                    'parameters': parameters,
                    'output': '',
                    'error': ''}

        db_connector = JobProcessor.get_db_connector(parameters)

        db_connector.write_job(new_line)

        python_command, python_path = cls._get_path_command()

        with open(out_file_name, "w") as f_out:
            with open(err_file_name, "w") as f_err:
                p = subprocess.Popen([python_command,
                                       python_path,
                                       '-background_job',
                                       new_job_id,
                                       '.'.join([module_name, function_name]),
                                       db_connector.db_id], stdout=f_out, stderr=f_err)

        return {'status': 'OK', 'error_text': '',
                'description': 'background job "{}" is started'.format('.'.join([module_name, function_name]))}

    @classmethod
    def _get_path_command(cls):

        python_command = 'python'
        python_path = 'job_processor.py'

        if sys.platform == "linux" or sys.platform == "linux2":
            python_command = 'python3'
            python_path = os.path.join('VectorModel', python_path)

        return python_command, python_path

    @classmethod
    def get_db_connector(cls, parameters):
        return DbConnector(parameters, initialize=True)

    @classmethod
    def get_jobs_state(cls, job_filter, parameters):

        db_connector = cls.get_db_connector(parameters)
        job_lines = db_connector.read_jobs(job_filter)

        job_lines_result = list()
        if job_lines:

            for job_line in job_lines:

                job_line_result = job_line.copy()

                if job_line_result.get('start_date'):
                    job_line_result['start_date'] = job_line_result['start_date'].strftime('%d.%m.%Y %H:%M:%S')
                if job_line_result.get('finish_date'):
                    job_line_result['finish_date'] = job_line_result['finish_date'].strftime('%d.%m.%Y %H:%M:%S')

                if job_line_result.get('parameters'):
                    job_line_result.pop('parameters')
                job_lines_result.append(job_line_result)

                if job_line['status'] in ['created', 'started']:

                    logger = StdOutErrLogger(job_line['job_id'], db_connector=db_connector)
                    out, err = logger.read_logs()

                    job_line['output'] = out
                    job_line['error'] = err

                    db_connector.write_job(job_line)

        return {'status': 'OK',
                'error_text': '',
                'description': 'jobs status received',
                'jobs': job_lines_result}

    @classmethod
    def get_job_state(cls, job_id, parameters):

        job_filter = dict(job_id=job_id)
        result = cls.get_jobs_state(job_filter,parameters)

        if result['jobs']:
            job_state_line = result['jobs'][0]
            result.pop('jobs')
            result['job_state'] = job_state_line
        else:
            result['job_state'] = {'status': 'does not exist'}

        result['description'] = 'job id {} status received'.format(job_id)
        return result

    @classmethod
    def delete_jobs(cls, del_filter, parameters):
        db_connector = cls.get_db_connector(parameters)
        db_connector.delete_jobs(del_filter)

        return {'status': 'OK',
                'error_text': '',
                'description': 'jobs deleted successfully'}


class StdOutErrLogger:

    def __init__(self, job_id, db_connector=None, parameters=None):
        self.job_id = job_id

        if db_connector:
            self._db_connector = db_connector
        elif parameters:
            self._db_connector = DbConnector(parameters)

        self._dir_name = 'job_logs'
        if not os.path.isdir(self._dir_name):
            os.mkdir(self._dir_name)

    def clear_old_logs(self):

        if not self._db_connector:
            return

        file_list = os.listdir(self._dir_name)

        for file_path in file_list:
            c_job_id = file_path[4:-4]

            os.remove(os.path.join(self._dir_name, file_path))

    def read_logs(self):

        out_file_name, err_file_name = self.get_log_file_names()

        out = ''
        if os.path.isfile(out_file_name):
            with open(out_file_name, 'r') as f:
                out = f.read()
        err = ''
        if os.path.isfile(err_file_name):
            with open(err_file_name, 'r') as f:
                err = f.read()

        return out, err

    def get_log_file_names(self):
        out_file_name = 'out_' + self.job_id + '.log'
        out_file_name = os.path.join(self._dir_name, out_file_name)
        err_file_name = 'err_' + self.job_id + '.log'
        err_file_name = os.path.join(self._dir_name, err_file_name)

        return out_file_name, err_file_name


def execute_method(system_parameters):

    try:
        job_id = str(uuid.UUID(system_parameters[2]))

        db_connector = JobProcessor.get_db_connector({'db_id': system_parameters[4]})

        if not db_connector:
            return {'status': 'error', 'error_text': 'db connector is not created'}

        job_line = db_connector.read_job(job_id)
    except Exception:

        error_text = traceback.format_exc()
        sys.stderr.write(error_text)

        return {'status': 'error', 'error_text': error_text}

    if not job_line or job_line['status'] != 'created':
        return {'status': 'error', 'error_text': 'job line not found'}
    try:

        job_line['status'] = 'started'
        db_connector.write_job(job_line)

        module_name, function_name = tuple(system_parameters[3].split('.'))

        imported_module = import_module(module_name)
        print(imported_module)
        imported_function = imported_module.__dict__[function_name]
        print(imported_function)
        function_parameters = job_line['parameters'].copy()
        function_parameters['background_job'] = False

        function_parameters['job_id'] = job_id

        result = imported_function(function_parameters)

        logger = StdOutErrLogger(job_id)
        out, err = logger.read_logs()

        job_line['finish_date'] = datetime.now()
        job_line['status'] = 'finished'
        job_line['result'] = result

        job_line['output'] = out
        job_line['error'] = err

        db_connector.write_job(job_line)

    except Exception:

        error_text = traceback.format_exc()
        sys.stderr.write(error_text)

        job_line['status'] = 'error'
        job_line['error'] = error_text

        db_connector.write_job(job_line)

        result = {'status': 'error', 'error_text': error_text}

    return result


def job_get_state(parameters):
    processor = JobProcessor(parameters)
    if parameters.get('job_id'):
        result = processor.get_job_state(parameters['job_id'], parameters)
    else:
        result = processor.get_jobs_state(parameters['filter'], parameters)

    return result


def job_delete(parameters):
    processor = JobProcessor(parameters)
    del_filter = parameters.get('filter')
    result = processor.delete_jobs(del_filter, parameters)

    return result

if __name__ == '__main__':

    if len(sys.argv) == 5:
        if sys.argv[1] == '-background_job':

            result = None
            error_text = ''
            try:
                result = execute_method(sys.argv)
                if result.get('error_text'):
                    error_text = result['error_text']
            except ProcessorException as exc:
                error_text = exc.get_msg()
            except Exception:
                error_text = traceback.format_exc()

            if error_text:
                job_id = str(uuid.UUID(sys.argv[2]))
                db_connector = JobProcessor.get_db_connector({'db_id': sys.argv[4]})
                job_line = db_connector.read_job(job_id)
                job_line['status'] = 'error'
                job_line['error'] = error_text





