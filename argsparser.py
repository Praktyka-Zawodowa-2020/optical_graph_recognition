import os
from shared import Mode
import argparse

parser = argparse.ArgumentParser("Optical graph recognition")
parser.add_argument("-p", "--path", help="Path to file")
parser.add_argument("-b", "--background",
                    help='''
                            GRID_BG - Hand drawn on grid/lined piece of paper (grid/lined notebook etc.) 
                            CLEAN_BG - Hand drawn on empty uniform color background 
                            PRINTED - Printed (e.g. from paper, publication, book...)
                            AUTO - Mode is chosen automatically
                        ''',
                    default='CLEAN_BG',
                    choices=['CLEAN_BG', 'GRID_BG', 'PRINTED', 'AUTO']
                    )


def parse_argument(args) -> (int, str, str):
    save_path = parse_path(args.path)
    mode = Mode.get_mode(args.background)

    return mode, args.path, save_path


def parse_path(file_path: str) -> str:
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

