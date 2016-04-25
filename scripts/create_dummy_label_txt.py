import sys
import os

def main():
    if len(sys.argv) != 2:
        print("Usage: target_dir")
        exit()

    root_dir = os.getcwd()
    target_dir = root_dir + "/" + sys.argv[1]
    # Write to respective set file
    set_filename =  "./dummy_label.txt"
    f = open(set_filename, 'wt')
    
    filename_list = os.listdir(target_dir)
    for filename in filename_list:
        f.write(target_dir + filename + " 0" + "\n")

    f.close()

if __name__ == "__main__":
    main()
