import os

if __name__ == '__main__':
    path = '/home/gfeng/gfeng_ws/exp2'
    f = open(path + '/eval/classification_report.txt', 'a')
        #import pdb; pdb.set_trace(
    f.write(path)
    f.write("\n")
    f.write("{}".format(path))
    f.write("{}".format(path))
    f.close()
    