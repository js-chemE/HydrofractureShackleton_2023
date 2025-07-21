import app.geotiffs as geotiffs
from app.config_handling import *
from app.dates import *
from app.drainages import *
from app.tides import *
import app.dmg as dmg 
import app.files as files
from app.logging import setup_logger

# setup_logger()

ENVIRONMENT = read_config()
