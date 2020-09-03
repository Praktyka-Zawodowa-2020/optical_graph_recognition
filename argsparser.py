import os
import argparse

from shared import Mode, Debug

# instance of parser for reading cli arguments
parser = argparse.ArgumentParser("Optical graph recognition")
parser.add_argument("-p", "--path", help="Path to file")
parser.add_argument("-m", "--mode", help=Mode.HELP, choices=Mode.CHOICES, default=Mode.DEFAULT, type=str.lower)
parser.add_argument("-d", "--debug", help=Debug.HELP, choices=Debug.CHOICES, default=Debug.DEFAULT, type=str.lower)


def parse_argument(args) -> (int, str, str):
    """
    Parses the command line arguments

    :param: args: Command line arguments
    :return: mode, path to photo, path to save the result
    """
    save_path = parse_path(args.path)
    mode = Mode.get_mode(args.mode)
    debug = Debug.get_debug(args.debug)

    return mode, debug, args.path, save_path


def parse_path(file_path: str) -> str:
    """
    Checks the path to the photo and specifies the path to save

    :param: file_path: path to photo
    :return: path to save the result

    """
    file_path.replace(" ", "")
    if file_path.count('.') != 1:
        print("1: File path is incorrect. Must be only one dot.")
        return ''
    head, tail = os.path.split(file_path)
    if len(tail) == 0:
        print("1: File name no exist")
        return ''

    file_name, file_ext = os.path.splitext(tail)
    if len(file_name) == 0:
        print("1: File name not found")
        return ''
    save_path = head + '/' + file_name
    return save_path
