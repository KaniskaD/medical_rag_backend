def safe_dict(val):
    return val if isinstance(val, dict) else {}

def safe_list(val):
    return val if isinstance(val, list) else []

def safe_number(val, default=0):
    return val if isinstance(val, (int, float)) else default

def safe_str(val, default=""):
    return val if isinstance(val, str) else default
