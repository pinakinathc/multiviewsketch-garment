""" Measures the diff between non-overlapping difference in SSIM """
import numpy as np

# data = open('result_non_overlap.txt').read().split('\n')[:-1]
data = open('result_overlap.txt').read().split('\n')[:-1]

all_ssim_diff = []

# for view 180 or front
ssim_front = data[1::3]
for i in range(0, len(ssim_front)-1, 2):
    new_ssim = float(ssim_front[i+1].split()[-1])
    old_ssim = float(ssim_front[i].split()[-1])
    all_ssim_diff.append(new_ssim - old_ssim)

all_ssim_diff = np.array(all_ssim_diff)
print ('Mean: {}, Median: {}, Max: {}, Min: {}'.format(
    np.mean(all_ssim_diff), np.median(all_ssim_diff), np.max(all_ssim_diff), np.min(all_ssim_diff)))


all_ssim_diff = []
# for view 0 or back
ssim_back = data[2::3]
for i in range(0, len(ssim_back)-1, 2):
    new_ssim = float(ssim_back[i+1].split()[-1])
    old_ssim = float(ssim_back[i].split()[-1])
    all_ssim_diff.append(new_ssim - old_ssim)

all_ssim_diff = np.array(all_ssim_diff)
print ('Mean: {}, Median: {}, Max: {}, Min: {}'.format(
    np.mean(all_ssim_diff), np.median(all_ssim_diff), np.max(all_ssim_diff), np.min(all_ssim_diff)))

# For chamfer distance
all_chamfer_dist = data[0::3]
diff = []
for i in range(0, len(all_chamfer_dist)-1, 2):
    new_dist = float(all_chamfer_dist[i+1].split()[-1])
    old_dist = float(all_chamfer_dist[i].split()[-1])
    diff.append(new_dist - old_dist)
diff = np.array(diff)

print ('Mean: {}, Median: {}, Max: {}, Min: {}'.format(
    np.mean(diff), np.median(diff), np.max(diff), np.min(diff)))
