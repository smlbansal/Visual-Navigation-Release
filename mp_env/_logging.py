import logging
log_formatter = logging.Formatter('%(asctime)s: %(levelname)s %(process)d %(filename)s:%(lineno)d] %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logging.getLogger().handlers = []
if not len(logging.getLogger().handlers): logging.getLogger().addHandler(console_handler)
logging.getLogger().setLevel(logging.INFO)
