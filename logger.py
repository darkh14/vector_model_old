
class ProcessorException(Exception):
    def __init__(self, msg='', **kwargs):
        super().__init__()
        self.msg = self.get_msg(msg)

    def get_msg(self, msg=None):
        return 'Error!' + ' ' + msg if msg else ''

    def __str__(self):
        return self.msg

