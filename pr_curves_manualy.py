

# !/usr/bin/env python

import tables as tb
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import sys

from teca.analytics.metrix import pr_curve, pr_curve_macro, reclev11_max, contingency_table, auc, seq_contingency_table, precision_recall_scores

# 7Genres
fig_save_file = '/home/dimitrios/TempResults/pr_curve_micro_example.eps'

plt_dsp_attr = [
    # ['grey', '--', '+', "OCSVM - TF"],
    # ['black', '--', '^', "RFSE - TF"],
    ['red', '-', '*', ""],
    ['grey', '--', '+', "NNDR - DF"],
]


p1 = [1]*13 + [2]
p2 = [2]*10 + [1]
p3 = [4]*8 + [5, 5, 6, 6]
p5 = [5]*20
p6 = [6]*10 + [2, 5, 7]
p7 = [7]*8 + [2, 2, 2, 3, 3, 3, 4, 5]

pred_y = \
    p1 + [0] * (20 - len(p1)) +\
    p2 + [0] * (20 - len(p2)) +\
    p3 + [0] * (20 - len(p3)) +\
    p4 + [0] * (20 - len(p4)) +\
    p5 + [0] * (20 - len(p5)) +\
    p6 + [0] * (20 - len(p6)) +\
    p7 + [0] * (20 - len(p7))

pred_y = np.array(pred_y, dtype=np.int32)
 
expd_y = np.hstack([[tg+1]*20 for tg in range(7)])
expd_y = np.array(expd_y, dtype=np.int32)

# pred_y = np.array([1,2,2,1,1,2,3,2,2,2,3,3,4,3,3,4,4,4,4,4])
# expd_y = np.array([1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4])
# pred_scores = np.arange(0.99, 0.00, -0.05)

print pred_y, len(pred_y)
print expd_y, len(expd_y)

pred_scores = \
    np.hstack((
        np.arange(0.99, 0.1, -0.05)[0:13], np.array([0.1]), 
        np.array([0.20] * (20 - len(p1))),
      
        np.arange(0.99, 0.1, -0.05)[0:10], np.array([0.3, 0.01]), 
        np.array([0.01] * (20 - len(p2))),

        np.array([0.77]), np.array([0.3, 0.2, 0.1, 0.3, 0.2, 0.1, 0.3, 0.2, 0.1, 0.3]), 
        np.array([0.01] * (20 - len(p3))),

        np.arange(0.99, 0.1, -0.05)[0:8], np.array([0.1, 0.3, 0.01, 0.3]), 
        np.array([0.60] * (20 - len(p4))),

        np.arange(0.99, 0.01, -0.05)[0:20],

        np.arange(0.99, 0.1, -0.05)[0:10], np.array([0.1, 0.2, 0.1]), 
        np.array([0.80] * (20 - len(p6))),

        np.arange(0.99, 0.1, -0.05)[0:8], np.array([0.3, 0.2, 0.1, 0.3, 0.01, 0.4, 0.1, 0.01]), 
        np.array([0.01] * (20 - len(p7)))
    ))
    
# pred_scores = np.hstack(pred_scores)

print pred_scores, len(pred_scores)

# print pred_scores, len(pred_scores)

# Shorting by score
inv_srd_idx = np.argsort(pred_scores)[::-1]
pred_y = pred_y[inv_srd_idx]
expd_y = expd_y[inv_srd_idx]
pred_scores = pred_scores[inv_srd_idx]
print pred_y, len(pred_y)
print expd_y, len(expd_y)
print pred_scores, len(pred_scores)

# # # #  The Ploting Process Starts Here # # # #
fig = plt.figure(num=1, figsize=(12, 8), facecolor='w', edgecolor='k')  # dpi=300,
ax = fig.add_subplot(111)

# annots_lst = list()
# labels_lst = list()

cm =  contingency_table(expd_y, pred_y, unknown_class=True)
print cm

# Creating the Actual PRC.
# y, x, t = pr_curve(expd_y, pred_scores, full_curve=True, is_truth_tbl=True)

# Creating the Actual MACRO PRC.

y1, x1, t, srz = pr_curve_macro(
    expd_y, pred_y, pred_scores, full_curve=True
)

yy, xx = y1, x1, 

# print srz, len(srz) 
# print t, len(t)
# print pred_scores, len(srz)
# print y1, len(y1)

trh_arr = np.where((expd_y == pred_y), 1, 0)
y2, x2, t = pr_curve(trh_arr, pred_scores, full_curve=True)

yy2, xx2 = y2, x2

# y, x, srz = y[4::], x[4::], srz[4::]

# print y, x,

# for ii, ss in enumerate(srz):
#    print ii, ss

# Getting the max 11 Recall Leves in TREC way.
# if i == 0:
y1, x1 = reclev11_max(y1, x1, trec=False)
y2, x2 = reclev11_max(y2, x2, trec=False)
# print
# print y, x

print 'AUC', auc(xx, yy)
print 'AUC MICRO', auc(xx2, yy2)

# Getting the number of samples per class. Zero tag is inlcuded.
smpls_per_cls = np.bincount(np.array(expd_y, dtype=np.int))

# Keeping from 1 to end array in case the expected class tags start with above zero values.
if smpls_per_cls[0] == 0 and np.unique(expd_y)[0] > 0:
    smpls_per_cls = smpls_per_cls[1::]
elif smpls_per_cls[0] > 0 and np.unique(expd_y)[0] == 0:
    pass  # same as --> smpls_per_cls = smpls_per_cls
    # Anythig else should rase an Exception.
else:
    raise Exception("Samples count in zero bin is different to the expected class tag cnt!")
print 'SAMPLES', smpls_per_cls, len(expd_y)
conf_mtrx = seq_contingency_table(
    expd_y, pred_y, exp_cls_tags_set=smpls_per_cls, arr_type=np.int32
)

# Calculating Precision per class.
precisions_vect = [
    dg / float(pred_docs)
    for dg, pred_docs in zip(np.diag(conf_mtrx), np.sum(conf_mtrx, axis=1))
    if pred_docs > 0
]

# Calculating Recall per class.
recalls_vect = [
    dg / float(splpc)
    for dg, splpc in zip(np.diag(conf_mtrx), smpls_per_cls)
    if splpc > 0
]

# NOTE: The Precision and Recall vectors of scores per Genre might not have the same...
# ...leght due to Unknown_Class tag expected or not expected case or just because...
# ...for some Class we have NO Predicitons at all.
macro_p = np.mean(precisions_vect)
macro_r = np.mean(recalls_vect)

print 'F1', 2.0 * macro_p*macro_r / (macro_p+macro_r)


print 'F1 MICRO', 2.0 * 0.714 * 0.500 / (0.714 + 0.500)

# Selecting array indices with non-zero cells.
non_zero_idx1 = np.where(y1 > 0)
non_zero_idx2 = np.where(y2 > 0)

# # # Do the Plotting
linestyle1 = {
    "color": plt_dsp_attr[0][0],
    "linestyle": plt_dsp_attr[0][1],
    "marker": plt_dsp_attr[0][2],
    "linewidth": 2,
    "markeredgewidth": 6,
    'markeredgecolor': plt_dsp_attr[0][0],
}

linestyle2 = {
    "color": plt_dsp_attr[1][0],
    "linestyle": plt_dsp_attr[1][1],
    "marker": plt_dsp_attr[1][2],
    "linewidth": 2,
    "markeredgewidth": 6,
    'markeredgecolor': plt_dsp_attr[1][0],
}

ax.plot(x1[non_zero_idx1], y1[non_zero_idx1], **linestyle1)
ax.plot(x2[non_zero_idx2], y2[non_zero_idx2], **linestyle2)

# annots_lst.append(mlines.Line2D([], [], markersize=0, linewidth=3, color=plt_dsp_attr[0][0]))
# labels_lst.append(plt_dsp_attr[0][3])

# lndump = mlines.Line2D([], [], markersize=0, linewidth=0)

"""
ax.annotate(
'F1=0.782, AUC=0.843',
xy=(0.12, 0.98), xytext=(0.2, 0.85), fontsize=16,
arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
bbox={'boxstyle':'round,pad=0.5','facecolor':'lightgray', 'alpha':0.9}
)
"""

# Give the poper attributes for better ploting
ax.yaxis.grid()

# lndump = mlines.Line2D([], [], markersize=0, linewidth=0)
# annots_lst.append(lndump)
# annots_lst.append(lndump)
# labels_lst.append("")
# labels_lst.append("")

"""
plt.legend(
    annots_lst,
    labels_lst,
    bbox_to_anchor=(0.0, 1.0, 1.0, 0.00),
    loc=3, ncol=3, mode="expand", borderaxespad=0.0,
    fancybox=False, shadow=False, fontsize=14
).get_frame().set_linewidth(0.0)
"""

# plt.yticks(fontsize=16)
plt.xticks(np.arange(0.0, 1.01, 0.1), fontsize=16)
plt.yticks(np.arange(0.45, 1.001, 0.05), fontsize=16)
plt.ylabel('Precision', fontsize=18)
plt.xlabel('Recall', fontsize=18)
plt.tight_layout()

# Saving the ploting to File
# plt.savefig(fig_save_file, bbox_inches='tight')

# plt.show()
