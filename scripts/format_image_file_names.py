import os

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def main():
    """ Put enough prepending 0 padding according to the README """
    input_dir = "./"

    for input_file in os.listdir(input_dir):
        filename, file_extension = os.path.splitext(input_file)
        if is_int(filename):
            new_filename = "{num:05d}".format(num=int(filename))
            os.rename(input_dir + input_file, input_dir + new_filename+file_extension)
        else:
            print(input_file + " does not have an number filename")


if __name__ == "__main__":
    main()
