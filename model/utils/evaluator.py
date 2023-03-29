import torch
import torch.nn.functional as F
import numpy as np


def cuda_dist(x, y):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    length = x.size(1)
    distance = list()
    weight = [1.0201, 1.0051, 1.2902, 1.3514, 1.0000, 1.1020, 1.6023, 1.0118,
              1.0515, 1.2688, 1.4982, 1.6145, 1.7124, 1.8715, 2.0000, 1.9694,
              1.0201, 1.0051, 1.2902, 1.3514, 1.0000, 1.1020, 1.6023, 1.0118,
              1.0515, 1.2688, 1.4982, 1.6145, 1.7124, 1.8715, 2.0000, 1.9694]
    for i in range(length):
        a = x[:, i, :]
        b = y[:, i, :]
        dist = torch.sum(a ** 2, 1).unsqueeze(1) + torch.sum(b ** 2, 1).\
            unsqueeze(1).transpose(0, 1) - 2 * torch.matmul(a, b.transpose(0, 1))
        dist = torch.sqrt(torch.sqrt(F.relu(dist)))
        if i > 14:
            dist = dist * weight[i-15]
        distance.append(dist)
    output = torch.stack(distance, 0).mean(0)
    return output


def evaluation(data, config):
    dataset = config['dataset'].split('-')[0]
    feature, view, seq_type, label = data
    label = np.array(label)
    view_list = list(set(view))
    view_list.sort()
    view_num = len(view_list)
    sample_num = len(feature)

    probe_seq_dict = {'CASIA': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'Infrared': [['fn02', 'fn03'], ['fb00', 'fb01'], ['fq00', 'fq01'], ['fs00', 'fs01']],
                      'OUMVLP': [['00']]}
    gallery_seq_dict = {'CASIA': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'Infrared': [['fn00', 'fn01']],
                        'OUMVLP': [['01']]}

    num_rank = 5
    acc = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num, num_rank])
    test_loss = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num, num_rank])
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for gallery_seq in gallery_seq_dict[dataset]:
            for (v1, probe_view) in enumerate(view_list):
                for (v2, gallery_view) in enumerate(view_list):
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(view, [gallery_view])
                    gallery_x = feature[gseq_mask, :]
                    gallery_y = label[gseq_mask]

                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view])
                    probe_x = feature[pseq_mask, :]
                    probe_y = label[pseq_mask]

                    dist = cuda_dist(probe_x, gallery_x)
                    probe_len = probe_y.shape[0]
                    gallery_len = gallery_y.shape[0]
                    probe_label = probe_y[:, np.newaxis]
                    gallery_label = gallery_y[:, np.newaxis].transpose((1, 0))
                    truth_mask = (np.repeat(probe_label, gallery_len, axis=1) == np.repeat(gallery_label,
                                                                                           probe_len, axis=0))
                    dist_array = dist.cpu().numpy()
                    positive_dist = dist_array[truth_mask]
                    min_mask = truth_mask + 0
                    min_dist = (dist_array + min_mask).min(1)[0]
                    test_loss[p, v1, v2, :] = np.round(np.mean(positive_dist - min_dist) + 0.2, 4)
                    idx = dist.sort(1)[1].cpu().numpy()
                    acc[p, v1, v2, :] = np.round(
                        np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                               0) * 100 / dist.shape[0], 2)

    return acc, test_loss
