import os
import os.path
from glob import glob

with open("aclimdb.txt", "wt") as outfile:
    for split in ['test', 'train']:
        for lbl in ['pos', 'neg']:
            for filename in glob(os.path.join("./aclImdb/", split, lbl, "*")):
                if os.path.isdir(filename) or not filename.endswith(".txt"):
                    continue
                with open(filename, "rt") as infile:
                    content = infile.read().strip()
                bname = "%s-%s-%s" % ( \
                        split, \
                        lbl, \
                        ".".join(os.path.basename(filename).split(".")[:-1])
                        )

                content = content.replace("\r", "\n").replace("\n\n", "\n").replace("\n", " ").replace("\t", " ")

                print(bname)
                outfile.write("%s\t%s\n" % (bname, content))



