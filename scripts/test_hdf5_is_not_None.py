import os
import h5py
import numpy as np

def check_and_delete_empty_hdf5_files(directory, dataset_name):
    # 遍历目录及其子目录
    count = 0
    count_empty = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.hdf5'):
                file_path = os.path.join(root, file)
                try:
                    with h5py.File(file_path, 'r') as f:
                        # 检查文件中是否包含特定的数据集
                        if dataset_name in f:
                            data = f[dataset_name]
                            if data.size > 0:
                                print(f"{file_path} contains data in {dataset_name}.")
                                count += 1
                            else:
                                print(f"-----------------{file_path} has an empty dataset {dataset_name}. Deleting file.")
                                f.close()  # 关闭文件后才能删除
                                count_empty += 1
                                # os.remove(file_path)
                        else:
                            print(f"-------------------{file_path} does not contain the dataset {dataset_name}. Deleting file.")
                            f.close()  # 关闭文件后才能删除
                            # os.remove(file_path)
                            count_empty += 1
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    print(f"Total files detected: {count}")
    print(f"Total empty files detected: {count_empty}")

# 指定要检查的目录和数据集名称
# directory = '/mnt/hpfs/baaiei/robot_data/pika/scoop_bean/红色杯子/杯子右/粉色勺子/episode26'
# dataset_name = '/eef_pose_L/xyz'
# 立明指定要检查的目录和数据集名称
directory = '/mnt/hpfs/baaiei/robot_data/realman/rm_stack_baskets/task_put_brown_black_basket/yellow_down'
dataset_name = '/observation/images'
check_and_delete_empty_hdf5_files(directory, dataset_name)