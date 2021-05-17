import os
import shutil
import tensorflow as tf
from datetime import datetime
from datetime import timedelta
import Augmentor

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.compat.v1.Session(config=config)


def augment(path):
    start = datetime.now().strftime("%H:%M:%S")
    all_folders = [x[0] for x in os.walk(path)]
    del all_folders[0]
    for folder in all_folders:
        p = Augmentor.Pipeline(folder)
        p.random_distortion(probability=1, grid_width=10, grid_height=10, magnitude=10)
        p.resize(probability=1, width=224, height=224)
        p.rotate(probability=0.7, max_left_rotation=25, max_right_rotation=25)
        p.skew(probability=0.9, magnitude=0.7)
        p.flip_random(probability=0.7)
        if "train" in path:
            p.sample(1000)
        elif "test" in path:
            p.sample(200)

        source = folder + "\\output"
        target = folder
        all_files = os.listdir(source)
        for file in all_files:
            shutil.move(os.path.join(source, file), target)
        os.rmdir(folder + "\\output")

    end = datetime.now().strftime("%H:%M:%S")
    start_to_end = datetime.strptime(end, "%H:%M:%S") - datetime.strptime(start,
                                                                          "%H:%M:%S")

    if start_to_end.days < 0:
        start_to_end = timedelta(days=0,
                                 seconds=start_to_end.seconds,
                                 microseconds=start_to_end.microseconds)

    if not os.path.exists("logs"):
        os.makedirs("logs")
    log_file = open("logs/augmentLogs.txt", "a")
    log_file.write("\nStart of the " + path + " augmentation: " + start +
                   "\nTime material augmentation was finished: " + end +
                   "\nTime it took to augment images: " + str(start_to_end))
    log_file.close()


augment("valladSeelikud\\train")
augment("valladSeelikud\\test")
augment("valladSeelikudVööd\\train")
augment("valladSeelikudVööd\\test")
augment("maakonnadSeelikud\\train")
augment("maakonnadSeelikud\\test")
augment("maakonnadSeelikudVööd\\train")
augment("maakonnadSeelikudVööd\\test")
augment("regioonidSeelikud\\train")
augment("regioonidSeelikud\\test")
augment("regioonidSeelikudVööd\\train")
augment("regioonidSeelikudVööd\\test")

