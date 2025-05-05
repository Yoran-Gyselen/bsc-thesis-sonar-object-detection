import sys

def parse_args(args):
    """
    Parses command-line arguments or a provided list of arguments in KEY=VALUE format.
    Returns a dictionary of config variables.
    """
    config = {}
    
    for arg in args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            try:
                config[key] = eval(value)
            except Exception:
                config[key] = value # Fallback to string if eval fails
        else:
            print(f"Ignoring invalid argument: {arg}")
    
    return config

if __name__ == "__main__":
    config = parse_args(sys.argv[1:])
    for key, value in config.items():
        print(f"{key} = {value}")