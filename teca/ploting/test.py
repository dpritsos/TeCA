import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

fig = plt.figure(num=1, figsize=(12, 7), facecolor='w', edgecolor='k')  # dpi=300,
ax = fig.add_subplot(111)


linestyle = {
    "color": 'red',
    "linestyle": '--',
    "marker": 'o',
    "linewidth": 2,
    "markeredgewidth": 2,
    'markeredgecolor': 'white',
}

ax.plot([1,2,3,4,5,6], [1,2,3,4,5,6], **linestyle)

annots_lst = [mlines.Line2D([], [], markersize=0, linewidth=3, color='red')]
labels_lst = ['test']


ax.yaxis.grid()

plt.legend(
    annots_lst,
    labels_lst,
    bbox_to_anchor=(0.0, 1.01, 1.0, 0.101),
    loc=3, ncol=3, mode="expand", borderaxespad=0.0,
    fancybox=False, shadow=False, fontsize=14
).get_frame().set_linewidth(0.0)

plt.yticks(fontsize=12)
plt.xticks(np.arange(0.0, 1.1, 0.1), fontsize=12)
plt.ylabel('Precision', fontsize=14)
plt.xlabel('Recall', fontsize=14)
# plt.tight_layout()

# Saving the ploting to File
# plt.savefig(fig_save_file, bbox_inches='tight')

plt.show()
