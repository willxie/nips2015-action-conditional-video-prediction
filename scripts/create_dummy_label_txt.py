import os

def main():
    root_dir = "/home/users/wxie/nips2015-action-conditional-video-prediction/pong_rand_actions/"

    # Write to respective set file
    set_filename =  "./dummy_label.txt"
    f = open(set_filename, 'wt')

    filename_list = os.listdir(root_dir)
    for filename in filename_list:
        f.write(root_dir + filename + " 0" + "\n")

    f.close()

if __name__ == "__main__":
    main()
