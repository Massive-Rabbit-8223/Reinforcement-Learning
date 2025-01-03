import matplotlib.pyplot as plt
import os
import pickle
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parent.parent)

PATH = os.path.dirname(os.path.abspath(__file__))
method_list = ["simple_pg", "reward_to_go_pg", "rtg_value_function_baseline_pg"]
metrics_dict = {}
for method in method_list:
    PATH_METRICS = ROOT_DIR + "/methods/"+method + "/training_statistics/metrics_dict_cpu_2.pkl"

    with open(PATH_METRICS, 'rb') as f:
        metrics = pickle.load(f)

    metrics_dict[method] = metrics

## Plot Mean Trajectory Lengths ##
for method, metrics in metrics_dict.items():
    plt.plot(metrics["mean_epoch_len"], label=method+": mean trajectory length")

plt.title("Average Number of Steps per Epoch")
plt.xlabel("Number of Epochs")
plt.ylabel("Steps")
plt.legend()
plt.savefig(PATH+"/plots/"+"mean_traj_len_2.pdf")
plt.savefig(PATH+"/plots/"+"mean_traj_len_2.jpg")
plt.show()

## Plot Epoch Loss ##
for method, metrics in metrics_dict.items():
    if "epoch_loss" in metrics.keys():
        plt.plot(metrics["epoch_loss"], label=method+": epoch loss")
    else:
        plt.plot(metrics["epoch_loss_policy"], label=method+": epoch loss")

plt.title("Loss per Epoch")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(PATH+"/plots/"+"epoch_loss_2.pdf")
plt.savefig(PATH+"/plots/"+"epoch_loss_2.jpg")
plt.show()