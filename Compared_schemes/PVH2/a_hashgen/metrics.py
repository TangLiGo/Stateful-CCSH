import matplotlib.pyplot as plt
def getAuc(labels, pred):
    '''将pred数组的索引值按照pred[i]的大小正序排序，返回的sorted_pred是一个新的数组，
       sorted_pred[0]就是pred[i]中值最小的i的值，对于这个例子，sorted_pred[0]=8
    '''
    sorted_pred = sorted(range(len(pred)), key=lambda i: pred[i])
    pos = 0.0  # 正样本个数
    neg = 0.0  # 负样本个数
    auc = 0.0
    last_pre = pred[sorted_pred[0]]
    count = 0.0
    pre_sum = 0.0  # 当前位置以前的预测值相等的rank之和，rank是从1开始的，因此在下面的代码中就是i+1
    pos_count = 0.0  # 记录预测值相等的样本中标签是正的样本的个数
    for i in range(len(sorted_pred)):
        if labels[sorted_pred[i]] > 0:
            pos += 1
        else:
            neg += 1
        if last_pre != pred[sorted_pred[i]]:  # 当前的预测几率值与前一个值不相同
            # 对于预测值相等的样本rank须要取平均值，而且对rank求和
            auc += pos_count * pre_sum / count
            count = 1
            pre_sum = i + 1  # 更新为当前的rank
            last_pre = pred[sorted_pred[i]]
            if labels[sorted_pred[i]] > 0:
                pos_count = 1  # 若是当前样本是正样本 ，则置为1
            else:
                pos_count = 0  # 反之置为0
        else:
            pre_sum += i + 1  # 记录rank的和
            count += 1  # 记录rank和对应的样本数，pre_sum / count就是平均值了
            if labels[sorted_pred[i]] > 0:  # 若是是正样本
                pos_count += 1  # 正样本数加1
    auc += pos_count * pre_sum / count  # 加上最后一个预测值相同的样本组
    auc -= pos * (pos + 1) / 2  # 减去正样本在正样本以前的状况
    auc = auc / (pos * neg)  # 除以总的组合数
    return auc


def getBenchmarks(tps, fps, fns, tns, thresholds, fig_path,range_flag=False):
    precision = []
    recall = []
    F1 = []
    iou = []
    accuracy = []
    for k in range(len(tps)):
        if tps[k] == 0:
            precision.append(0)
            recall.append(0)
            F1.append(0)
            iou.append(0)
            accuracy.append(0)
        else:
            precision.append(tps[k] / (tps[k] + fps[k]))
            recall.append(tps[k] / (tps[k] + fns[k]))
            F1.append(2 * tps[k] / (2 * tps[k] + fps[k] + fns[k]))
            iou.append(tps[k] / (tps[k] + fps[k] + fns[k]))
            accuracy.append((tps[k] + tns[k]) / (tps[k] + fps[k] + fns[k] + tns[k]))

    best_threshold_index = F1.index(max(F1))
    best_threshold = thresholds[best_threshold_index]

    print("best_threshold=", best_threshold)
    print("Precision={:.2%}".format(precision[best_threshold_index]))
    print("Recall={:.2%}".format(recall[best_threshold_index]))
    print("F1={:.2%}".format(F1[best_threshold_index]))
    print("IoU={:.2%}".format(iou[best_threshold_index]))
    print("Accuracy={:.2%}".format(accuracy[best_threshold_index]))
    plt.figure()
    plt.plot(thresholds, precision, label="precision")
    plt.plot(thresholds, recall, label="recall")
    plt.plot(thresholds, F1, label="F1")
    plt.plot(thresholds, iou, label="iou")
    plt.savefig(fig_path)
    print("tp:{},fp:{},fn:{},tn:{}".format(tps[best_threshold_index],fps[best_threshold_index],fns[best_threshold_index],tns[best_threshold_index]))
    return precision, recall, F1, iou, accuracy

