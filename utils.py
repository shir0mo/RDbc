import os
import datetime
import pytz

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

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
        filename = "../train_log/" + _class_ + "_" + get_day + "_" + get_time + ".txt"
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

# t-sne visualization
def plot_tsne(features, labels, epoch, cls_list, _class_, _path_):

    tsne_path = "../visualize/" + _class_
    if not os.path.exists(tsne_path):
        os.mkdir(tsne_path)
    else:
        print("directory is already exists.")
    
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)

    colors = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabebe", "#469990", "#e6beff", "#9A6324", "#800000"
    ]

    # mpdd_colors = [
    #     "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231"
    # ]

    custom_cmap = mcolors.ListedColormap(colors)

    plt.figure(figsize=(15, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap=custom_cmap, s=10)

    patches = [mpatches.Patch(color=custom_cmap(i), label=cls_list[i]) for i in range(len(cls_list))]
    plt.legend(handles=patches, title='Classes', bbox_to_anchor=(1., 1), loc='upper left')

    #plt.colorbar(scatter)
    plt.title(f't-SNE Visualization at Epoch {epoch}')
    plt.savefig(f'{tsne_path}/{_path_}_tsne_epoch_{epoch}.png')  # Save the plot for each epoch

    del(tsne_results)