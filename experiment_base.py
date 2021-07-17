import sys
import os
import datetime
import contextlib


@contextlib.contextmanager
def print_to_logfile(file):
    # capture all outputs to a log file while still printing it
    class Logger:
        def __init__(self, file):
            self.terminal = sys.stdout
            self.log = file

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def __getattr__(self, attr):
            return getattr(self.terminal, attr)

    logger = Logger(file)

    _stdout = sys.stdout
    sys.stdout = logger
    try:
        yield logger.log
    finally:
        sys.stdout = _stdout


def initiate_experiment(experiment):

    def decorator(*args, **kwargs):
        log_file_dir = "logs/"
        log_file = log_file_dir + experiment.__name__ + ".txt"
        if not os.path.exists(log_file):
            os.makedirs(log_file_dir, exist_ok=True)
            os.makedirs('plots/', exist_ok=True)
        with print_to_logfile(open(log_file, 'w')):
            print("Performing experiment:", experiment.__name__)
            print("Date-Time:", datetime.datetime.now())
            print("\n", end="")
            print("Args:", args)
            print("Kwargs:", kwargs)
            print("\n", end="")
            experiment(*args, **kwargs)
            print("\n\n", end="")
    return decorator
