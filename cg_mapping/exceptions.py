class CGMappingError(Exception):
    """ Base class for package's exceptions. """
    pass

class OutOfOrderError(CGMappingError):
    """ Raised when a function is called out of order """
    pass