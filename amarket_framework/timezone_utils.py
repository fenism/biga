"""
Timezone utility module for handling Beijing time (UTC+8).
Centralizes timezone handling to ensure consistent time operations across the application.
"""
import pytz
from datetime import datetime

# Beijing timezone constant (Asia/Shanghai = UTC+8)
BEIJING_TZ = pytz.timezone('Asia/Shanghai')


def get_beijing_now():
    """
    Get current time in Beijing timezone (UTC+8).
    
    Returns:
        datetime: Current datetime with Beijing timezone
    """
    return datetime.now(BEIJING_TZ)


def to_beijing_time(dt):
    """
    Convert a datetime object to Beijing timezone.
    
    Args:
        dt (datetime): Datetime object to convert (can be naive or aware)
    
    Returns:
        datetime: Datetime in Beijing timezone
    """
    if dt.tzinfo is None:
        # If naive datetime, assume it's already in Beijing time
        return BEIJING_TZ.localize(dt)
    else:
        # If aware datetime, convert to Beijing timezone
        return dt.astimezone(BEIJING_TZ)


def get_beijing_date_str(format='%Y-%m-%d'):
    """
    Get current Beijing date as formatted string.
    
    Args:
        format (str): Date format string (default: '%Y-%m-%d')
    
    Returns:
        str: Formatted date string in Beijing timezone
    """
    return get_beijing_now().strftime(format)


def get_beijing_datetime_str(format='%Y-%m-%d %H:%M:%S'):
    """
    Get current Beijing datetime as formatted string.
    
    Args:
        format (str): Datetime format string (default: '%Y-%m-%d %H:%M:%S')
    
    Returns:
    """
    return get_beijing_now().strftime(format)

def is_trading_time():
    """
    Check if current time is within A-share trading hours.
    Trading hours: Monday-Friday, 09:30-11:30, 13:00-15:00.
    """
    now = get_beijing_now()
    
    # Weekends
    if now.weekday() > 4:  # 0=Monday, 4=Friday, 5=Saturday, 6=Sunday
        return False
        
    current_time = now.time()
    
    # Morning session: 09:30 - 11:30
    import datetime
    morning_start = datetime.time(9, 30)
    morning_end = datetime.time(11, 30)
    
    # Afternoon session: 13:00 - 15:00
    afternoon_start = datetime.time(13, 0)
    afternoon_end = datetime.time(15, 0)
    
    if (morning_start <= current_time <= morning_end) or \
       (afternoon_start <= current_time <= afternoon_end):
        return True
        
    return False
