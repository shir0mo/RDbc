import os
import datetime
import pytz

# create log
def create_log_file(_class_):

    # get time
    dt_now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    get_day = str(dt_now.year) + "-" + str(dt_now.month) + "-" + str(dt_now.day)
    get_time = str(dt_now.hour) + "-"  + str(dt_now.minute)

    filename = "../train_log/" + _class_ + "_" + get_day + ".txt"

    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.close()
    else:
    # ファイル名が存在するなら，時間を付与
        filename = "../" + _class_ + "_" + get_day + "_" + get_time + ".txt"
        with open(filename, "w") as f:
            f.close()
        
    print("create {}".format(filename[3:]))
    
    return filename
    
# print function
def log_and_print(message, file="log.txt"):
    print(message)  # 標準出力

    assert os.path.exists(file), "{} is not exists.".format(file)

    with open(file, "a") as f:
        f.write(message + "\n")
        f.close()

