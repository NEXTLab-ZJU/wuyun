from datetime import datetime
from datetime import timedelta
from datetime import timezone
import time

def get_time():
    SHA_TZ = timezone(timedelta(hours=8), name='Asia/Shanghai',)
    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)   
     # Beijing
    beijing_now = utc_now.astimezone(SHA_TZ)
    beijing_now_str = beijing_now.strftime("%Y%m%d:%H%M%S")
    return beijing_now_str
    

get_time()