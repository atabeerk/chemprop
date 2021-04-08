import os
from math import ceil
from shutil import copyfile


THRESHOLD = 10000000

def split(file, dest_folder):
        file_size = os.stat(file).st_size
        if file_size > THRESHOLD:
            n_chunks = ceil(file_size / THRESHOLD)
            f = open(file, 'r')
            for i in range(n_chunks):
                new_name = "_".join([os.path.basename(file).split(".")[0], "chunk", str(i+1)]) + ".txt"
                lines = f.readlines(THRESHOLD)
                with open(os.path.join(dest_folder, new_name), "a+") as w:
                    w.writelines(lines)
            f.close()
        else:
            copyfile(file, os.path.join(dest_folder, os.path.basename(file)))


def split_folder(folder, dest_folder):
    for file in os.listdir(folder):
        split(os.path.join(folder, file), dest_folder)

if __name__ == "__main__":
    split_folder("../zinc_flagments", "splitted")