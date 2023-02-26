# Helper Function to value in nested dictionary (crawling through the json files)
def find(dict_name, key_path):
    """
    Find the value of a key_path separated by periods in a nested dictionary (or list).
    :param dict_name: The dictionary to search.
    :param key_path: The path to the key, separated by periods.
    :return: The value of the key.

    Example usage:
    d = {'a': {'b': {'c': [1,2,3]}}}
    find(d, 'a.b.c.1') # returns 2
    """

    keys = key_path.split('.')
    rv = dict_name
    for key in keys:
        if key.isdigit(): key = int(key)
        rv = rv[key]
    return rv