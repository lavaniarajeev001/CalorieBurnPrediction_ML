import os
from src.logger import logging

import sys

def error_mesage_details(error,error_detals:sys):
    _,_,exc_tb=error_detals.exc_info()

    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(file_name,exc_tb.tb_lineno,str(error))
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detals:sys):
        super().__init__(error_message)
        self.error_mesage=error_mesage_details(error_message,error_detals=error_detals)

    def __str__(self):
        return self.error_mesage