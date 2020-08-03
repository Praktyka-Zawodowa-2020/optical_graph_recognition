import os
from shared import Mode


def parse_argument(args) -> (int, str, str):
    save_path = parse_path(args.path)
    mode = parse_background(args.background)

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


def parse_background(mode: str) -> int:
    if mode == "GRID_BG":
        return Mode.GRID_BG
    elif mode == "CLEAN_BG":
        return Mode.CLEAN_BG
    elif mode == "PRINTED":
        return Mode.PRINTED
    else:
        print("1: Mode not found")
        return -1
