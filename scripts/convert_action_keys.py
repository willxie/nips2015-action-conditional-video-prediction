import os
import sys

def is_int(s):    
    try: 
        int(s)
        return True
    except ValueError:
        return False


def main(): 
    if len(sys.argv) != 2:
        print("Usage: target_dir")
        exit()

    root_dir = os.getcwd()
    input_file = root_dir + "/" + sys.argv[1] + "act.log"
    output_file = "./act.log"
    output_file_key = "./act_key.txt"
    num_actions = 0
    container = {}
    
    filename, file_extension = os.path.splitext(input_file)
    with open(input_file, "rb") as r, open(output_file, "wb") as w:
        for line in r:
            line_s = line.rstrip("\n")
            if (is_int(line_s)):
                action_num = int(line_s)
                if (action_num not in container):        
                    container[action_num] = num_actions
                    num_actions += 1
                w.write(str(container[action_num]))
                w.write("\n")
    
    
    # Write the keys
    with open(output_file_key, "wb") as f:
        f.write("orig" + "\t=>\t" + "new" + "\n")
        for key, value in container.iteritems():
            f.write(str(key) + "\t=>\t" + str(value) + "\n")
        f.write("Total number of actions: {}".format(num_actions))    

if __name__ == "__main__":
    main()
