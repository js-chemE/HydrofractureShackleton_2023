import logging
import os

#log_dir = os.path.expanduser(r"~\[Code]\c1L100-datamanager")

def setup_logger():
    logging.basicConfig(
        filename='app.log',
        filemode='w',
        encoding='utf-8',
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(name)s %(message)s'
    )