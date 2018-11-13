import logging, sys


class Logger(object):
    """
    A class to to write to two places at once--log file and the stdout.
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        
        
def setup_logger(filename):
    """
    Setup a logger that automatically prints the messages of stdout to the log file.
    """
    logging.basicConfig(level=logging.INFO, filename=filename, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    sys.stdout = Logger(filename)

    # To print the messages into the logger directly
    # logging.info('This message is coming from the logger directly.')
    # To print the messages into the logger through stdout.
    # print('This message is coming from stdout.')
