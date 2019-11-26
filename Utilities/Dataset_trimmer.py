import cv2
import shutil

def trim(num_train, num_val, num_test):
    print("Trimming training set")
    (start, end) = input().split(' ')
    print(start)
    num = 0
    i = 0
    ROIs = open("..\\Data_saver\\ROIs\\ROIs.txt")
    ROIs_trim = open("..\\Data_saver\\ROIs\\ROIs_trim.txt", "w")
    rects = [line.rstrip('\n') for line in ROIs]
    while i < int(num_train):
        print(i)
        if i == int(start):
            i = int(end) + 1
            print(str(i), "Skip until:")
            (start, end) = input().split(' ')
        shutil.move("..\\Data_saver\\color\\col" + str(i) + ".png", "..\\Data_saver\\color_trim\\col" + str(num) + ".png")
        shutil.move("..\\Data_saver\\depth_color\\col_depth" + str(i) + ".png", "..\\Data_saver\\depth_color_trim\\col_depth" + str(num) + ".png")
        shutil.move("..\\Data_saver\\depth_data\\depth" + str(i) + ".png", "..\\Data_saver\\depth_data_trim\\depth" + str(num) + ".png")
        ROIs_trim.write(rects[i])
        ROIs_trim.write('\n')
        num = num + 1
        i = i + 1

    master = open("..\\Data_saver\\master_trim.txt", "w")
    master.write(str(num))
    master.write('\n')

    print("Trimming validation set")
    (start, end) = input().split(' ')
    print(start)
    num = 0
    i = 0
    ROIs = open("..\\Data_saver\\ROIs\\val_ROIs.txt")
    ROIs_trim = open("..\\Data_saver\\ROIs\\val_ROIs_trim.txt", "w")
    rects = [line.rstrip('\n') for line in ROIs]
    while i < int(num_val):
        print(i)
        if i == int(start):
            i = int(end) + 1
            print(str(i), "Skip until:")
            (start, end) = input().split(' ')
        shutil.move("..\\Data_saver\\color\\vali_col" + str(i) + ".png", "..\\Data_saver\\color_trim\\vali_col" + str(num) + ".png")
        shutil.move("..\\Data_saver\\depth_color\\vali_col_depth" + str(i) + ".png", "..\\Data_saver\\depth_color_trim\\vali_col_depth" + str(num) + ".png")
        shutil.move("..\\Data_saver\\depth_data\\vali_depth" + str(i) + ".png", "..\\Data_saver\\depth_data_trim\\vali_depth" + str(num) + ".png")
        ROIs_trim.write(rects[i])
        ROIs_trim.write('\n')
        num = num + 1
        i = i + 1

    master.write(str(num))
    master.write('\n')

    print("Trimming test set")
    (start, end) = input().split(' ')
    num = 0
    i = 0
    ROIs = open("..\\Data_saver\\ROIs\\test_ROIs.txt")
    ROIs_trim = open("..\\Data_saver\\ROIs\\test_ROIs_trim.txt", "w")
    rects = [line.rstrip('\n') for line in ROIs]
    while i < int(num_test):
        print(i)
        if i == int(start):
            i = int(end) + 1
            print(str(i), "Skip until:")
            (start, end) = input().split(' ')
        shutil.move("..\\Data_saver\\color\\test_col" + str(i) + ".png", "..\\Data_saver\\color_trim\\test_col" + str(num) + ".png")
        shutil.move("..\\Data_saver\\depth_color\\test_col_depth" + str(i) + ".png", "..\\Data_saver\\depth_color_trim\\test_col_depth" + str(num) + ".png")
        shutil.move("..\\Data_saver\\depth_data\\test_depth" + str(i) + ".png", "..\\Data_saver\\depth_data_trim\\test_depth" + str(num) + ".png")
        ROIs_trim.write(rects[i])
        ROIs_trim.write('\n')
        num = num + 1
        i = i + 1

    master.write(str(num))

def dec(num_train, num_test):
    print("Decimating training set")
    (start, end) = input().split(' ')
    num = 0
    i = 0
    d = 0
    dec_factor = 3
    ROIs = open("..\\Data_saver\\ROIs\\ROIs.txt")
    ROIs_trim = open("..\\Data_saver\\ROIs\\ROIs_trim.txt", "w")
    rects = [line.rstrip('\n') for line in ROIs]
    while i < int(num_train):
        if i == int(end):
            (start, end) = input().split(' ')
        print(i)
        shutil.move("..\\Data_saver\\color\\col" + str(i) + ".png", "..\\Data_saver\\color_trim\\col" + str(num) + ".png")
        shutil.move("..\\Data_saver\\depth_color\\col_depth" + str(i) + ".png", "..\\Data_saver\\depth_color_trim\\col_depth" + str(num) + ".png")
        shutil.move("..\\Data_saver\\depth_data\\depth" + str(i) + ".png", "..\\Data_saver\\depth_data_trim\\depth" + str(num) + ".png")
        ROIs_trim.write(rects[i])
        ROIs_trim.write('\n')
        num = num + 1
        d = d + 1
        i = i + 1 if i < int(start) or d % dec_factor == 0  else i + 2

    master = open("..\\Data_saver\\master_dec.txt", "w")
    master.write(str(num))
    master.write('\n')
    print("Decimating test set")
    (start, end) = input().split(' ')
    num = 0
    i = 0
    ROIs = open("..\\Data_saver\\ROIs\\test_ROIs.txt")
    ROIs_trim = open("..\\Data_saver\\ROIs\\test_ROIs_trim.txt", "w")
    rects = [line.rstrip('\n') for line in ROIs]
    while i < int(num_test):
        print(i)
        shutil.move("..\\Data_saver\\color\\test_col" + str(i) + ".png", "..\\Data_saver\\color_trim\\test_col" + str(num) + ".png")
        shutil.move("..\\Data_saver\\depth_color\\test_col_depth" + str(i) + ".png", "..\\Data_saver\\depth_color_trim\\test_col_depth" + str(num) + ".png")
        shutil.move("..\\Data_saver\\depth_data\\test_depth" + str(i) + ".png", "..\\Data_saver\\depth_data_trim\\test_depth" + str(num) + ".png")
        ROIs_trim.write(rects[i])
        ROIs_trim.write('\n')
        num = num + 1
        i = i + 1 if i < int(start) else i + 2

    master.write(str(num))


master = open("..\\Data_saver\\master.txt")
num_train = master.readline()
num_val = master.readline()
num_test = master.readline()
master.close()

choice = input()

if choice == "trim":
    trim(num_train, num_val, num_test)
elif choice == "dec":
    dec(num_train, num_test)
