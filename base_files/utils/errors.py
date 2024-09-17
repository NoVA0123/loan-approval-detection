'''
NOTE: THIS FILE IS DIRECTED TOWARDS SPECIALIZED ERRORS THAT THIS
REPOSITORY CONTAINS. SOME ERRORS MAY OR MAY NOT WORK DEPENDING ON
SITUATION.
'''
# Out of memory Error
class notEnoughMemory(Exception):
    """
    Raised when the file size does not leave behind suitable amount
    of memory while working on the file.
    """
    pass

