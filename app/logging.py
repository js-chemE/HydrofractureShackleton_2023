import logging

def setup_logger(use_file_handler=True, use_console_handler=False):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if use_file_handler:
        file_handler = logging.FileHandler('app.log', mode='w', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(name)s %(message)s'))
        logger.addHandler(file_handler)
    if use_console_handler:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(name)s %(message)s'))
        logger.addHandler(console_handler)