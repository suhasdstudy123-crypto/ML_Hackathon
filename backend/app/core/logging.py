import json
import logging
from datetime import datetime, timezone

class JSONFormatter(logging.Formatter):
    def format(self, record):
        body={'at':datetime.now(timezone.utc).isoformat(),'level':record.levelname,'event':record.getMessage()}
        for k in ['request_id','job_id','trip_id','stage','duration_ms','model_version','candidate_count','error_code','method','path','status_code','metrics']:
            if hasattr(record,k): body[k]=getattr(record,k)
        if record.exc_info: body['traceback']=self.formatException(record.exc_info)
        return json.dumps(body)

def configure_logging():
    logger=logging.getLogger('pathsnap'); logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler=logging.StreamHandler(); handler.setFormatter(JSONFormatter()); logger.addHandler(handler)
    logger.propagate=False
