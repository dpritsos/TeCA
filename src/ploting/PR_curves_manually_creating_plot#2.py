
import matplotlib.pyplot as plt

X = [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 ]

"""
#SANTINIS - 11-AVG - Strata
snt_ocsvm_Y = [1.0, 1.0, 0.976141929626, 0.860311388969, 0.718630433083, 0.644534170628, 0.617862939835, 0.613784849644, 0.613784849644, 0.613784849644, 0.61497759819]
snt_rfse_cos_3w_Y = [1.0, 0.97316390276, 0.969599068165, 0.950774729252, 0.925587296486, 0.891225218773, 0.844622194767, 0.80593162775, 0.780958533287, 0.780958533287, 0.765427052975]
snt_rfse_cos_1w_Y = [1.0, 0.986691653728, 0.974489688873, 0.975040853024, 0.939043819904, 0.904086887836, 0.840994298458, 0.792474865913, 0.759176850319, 0.759176850319, 0.761651694775]
snt_rfse_cos_4c_Y = [1.0, 0.957857012749, 0.967549800873, 0.963654637337, 0.88259190321, 0.815001070499, 0.776693880558, 0.759780228138, 0.73388260603, 0.73388260603, 0.734285712242]
snt_rfse_minmax_3w_Y = [1.0, 0.993797957897, 0.984527885914, 0.966746628284, 0.946692049503, 0.921767771244, 0.880797564983, 0.843810081482, 0.811821460724, 0.811821460724, 0.782466173172]
snt_rfse_minmax_1w_Y = [1.0, 0.980868399143, 0.970863997936, 0.965071797371, 0.908563792706, 0.872706472874, 0.803740739822, 0.748519480228, 0.731933891773, 0.731933891773, 0.729807317257]
snt_rfse_minmax_4c_Y = [1.0, 0.952644228935, 0.967345237732, 0.971721231937, 0.901053667068, 0.852000296116, 0.79404258728, 0.76769131422, 0.736150622368, 0.736150622368, 0.738367319107]
snt_rfse_MIX_3w_Y = [1.0, 0.986897289753, 0.979567825794, 0.969351232052, 0.958274304867, 0.932668685913, 0.869807302952, 0.835049510002, 0.819253087044, 0.819253087044, 0.792962372303]
snt_rfse_MIX_1w_Y = [1.0, 0.971176862717, 0.968612372875, 0.97529232502, 0.922534286976, 0.832491099834, 0.777258038521, 0.761963367462, 0.761177122593, 0.761177122593, 0.754400312901]
snt_rfse_MIX_4c_Y = [1.0, 0.967351317406, 0.970527529716, 0.970436811447, 0.915727972984, 0.838920116425, 0.811105132103, 0.786636471748, 0.759653925896, 0.759653925896, 0.750611245632]
snt_fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/PR_Curves_SANTINIS_Baseline_11AVG_Strata.eps'
"""


symbol = ['o', 'v', '^', '*', '<', 's', '+', 'x', '>', 'H', '1', '2', '3', '4', 'D', 'h', '8', 'd', 'p', '.', ',']
line_type = ['--', '--', '--', '-', '-', '-', '--', '--', '--', '-', '-', '-', '--.', '-' , '-', '-', '-', '--', '--', '--', '--']

fig = plt.figure(num=1, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_subplot(111)


#=================================================================================================
# RFSE 
#=================================================================================================

g7_rfse_cos_3w_Y = [1.0, 0.980733931065, 0.970042347908, 0.979685902596, 0.983170628548, 0.984299898148, 0.985552370548, 0.980423569679, 0.966289520264, 0.92120808363, 0.864285707474]
g7_rfse_mix_3w_Y = [1.0, 0.992379367352, 0.974547088146, 0.982898950577, 0.985347509384, 0.985847473145, 0.987708389759, 0.982666909695, 0.961867451668, 0.909256339073, 0.865000009537]
g7_rfse_cos_1w_Y = [1.0, 0.988514125347, 0.990173757076, 0.994233250618, 0.991900444031, 0.981351077557, 0.964436948299, 0.941907405853, 0.874918580055, 0.874918580055, 0.795000016689]
g7_rfse_cos_4c_Y = [1.0, 1.0, 0.999475359917, 0.996441781521, 0.995299458504, 0.990267693996, 0.976649701595, 0.957946658134, 0.929076433182, 0.869693815708, 0.825714290142]
g7_rfse_mix_4c_Y = [1.0, 1.0, 0.998293042183, 0.993604779243, 0.989560067654, 0.983956217766, 0.976785600185, 0.962196111679, 0.936504244804, 0.88490319252, 0.828571438789]
g7_rfse_minmax_3w_Y = [1.0, 0.908599972725, 0.955847561359, 0.962127566338, 0.966561019421, 0.960567772388, 0.962256550789, 0.959122419357, 0.948229491711, 0.904445052147, 0.872857153416]


ki04_rfse_cos_1w_Y = [1.0, 0.974679350853, 0.956727325916, 0.913120627403, 0.887999236584, 0.84011977911, 0.71924495697, 0.71924495697, 0.71924495697, 0.71924495697, 0.588381767273]
ki04_rfse_minmax_3w_Y = [1.0, 0.885461091995, 0.929193735123, 0.909013748169, 0.871427357197, 0.798218905926, 0.676263749599, 0.676263749599, 0.676263749599, 0.676263749599, 0.57012450695]
ki04_rfse_minmax_1w_Y = [1.0, 0.930316865444, 0.949570953846, 0.926678597927, 0.886715114117, 0.839007675648, 0.76428258419, 0.674283921719, 0.674283921719, 0.674283921719, 0.621576786041]
ki04_rfse_minmax_4c_Y = [1.0, 0.94373780489, 0.906122684479, 0.881386220455, 0.840561926365, 0.812357664108, 0.691382408142, 0.691382408142, 0.691382408142, 0.691382408142, 0.563485503197]
ki04_rfse_mix_1w_Y = [1.0, 0.94056814909, 0.927180469036, 0.910919547081, 0.86400949955, 0.793197393417, 0.697702825069, 0.697702825069, 0.697702825069, 0.697702825069, 0.598340272903]


snt_rfse_cos_1w_Y = [1.0, 0.764723420143, 0.743766725063, 0.811161756516, 0.830892682076, 0.855104148388, 0.872797966003, 0.888376653194, 0.895590245724, 0.896429896355, 0.897909402847]
snt_rfse_cos_3w_Y = [1.0, 0.931079566479, 0.812419772148, 0.810686349869, 0.835493087769, 0.855023384094, 0.868845641613, 0.886297225952, 0.891916930676, 0.894766628742, 0.896080136299]
snt_rfse_minmax_3w_Y = [1.0, 0.981221735477, 0.958775579929, 0.930858135223, 0.918592333794, 0.914032936096, 0.914665579796, 0.922808587551, 0.92455971241, 0.92622667551, 0.926093280315]
snt_rfse_minmax_1w_Y = [1.0, 0.772948741913, 0.794674634933, 0.850244104862, 0.85799074173, 0.876019835472, 0.890346348286, 0.902496993542, 0.903228521347, 0.903612017632, 0.906397283077]
snt_rfse_minmax_4c_Y = [1.0, 0.739262938499, 0.780263245106, 0.839130580425, 0.850026965141, 0.870198369026, 0.885551214218, 0.898147523403, 0.899340867996, 0.899637520313, 0.902401208878]
snt_rfse_mix_3w_Y = [1.0, 0.982258021832, 0.969178617001, 0.960849702358, 0.955416321754, 0.952583909035, 0.946125328541, 0.937441468239, 0.929030895233, 0.924789190292, 0.922538280487]

# The baseline - OCSVME PRCs
g7_ocsvm_3w_Y = [1.0, 1.0, 0.997583389282, 0.985482096672, 0.941586136818, 0.890766561031, 0.742332100868, 0.742332100868, 0.742332100868, 0.742332100868, 0.59500002861]
g7_ocsvm_1w_Y = [1.0, 0.871262133121, 0.831683933735, 0.791367352009, 0.747086644173, 0.684091389179, 0.633607625961, 0.633607625961, 0.633607625961, 0.633607625961, 0.595714271069]
g7_ocsvm_4c_Y = [1.0, 0.971968829632, 0.860191583633, 0.785888433456, 0.763845145702, 0.759213924408, 0.760704755783, 0.709067046642, 0.709067046642, 0.709067046642, 0.64571428299]
ki04_ocsvm_3w_Y = [1.0, 0.676602244377, 0.632809042931, 0.4316393435, 0.4316393435, 0.4316393435, 0.4316393435, 0.4316393435, 0.4316393435, 0.4316393435, 0.268879681826] 
ki04_ocsvm_1w_Y = [1.0, 0.309113562107, 0.41513505578, 0.406616270542, 0.39465329051, 0.39465329051, 0.39465329051, 0.39465329051, 0.39465329051, 0.39465329051, 0.365975111723]
ki04_ocsvm_4c_Y = [1.0, 0.436228632927, 0.408268511295, 0.440019071102, 0.412177503109, 0.412177503109, 0.412177503109, 0.412177503109, 0.412177503109, 0.412177503109, 0.353526979685]
snt_ocsvm_3w_Y = [1.0, 0.30569010973, 0.241448864341, 0.341055870056, 0.415944218636, 0.474728405476, 0.500130653381, 0.500130653381, 0.500130653381, 0.500130653381, 0.500174224377]
snt_ocsvm_1w_Y = [1.0, 0.136777102947, 0.161714568734, 0.213254570961, 0.213254570961, 0.213254570961, 0.213254570961, 0.213254570961, 0.213254570961, 0.213254570961, 0.221689894795]
snt_ocsvm_4c_Y = [1.0, 0.167120173573, 0.276200115681, 0.384811013937, 0.46413937211, 0.520765542984, 0.556276738644, 0.556276738644, 0.556276738644, 0.556276738644, 0.568902432919]


"""
# 7Genres 3W for every Distance Measure
plt.plot(X, g7_rfse_cos_3w_Y, 'k' + line_type[1] + symbol[1], linewidth=1, markersize=9, label="3W - Cosine")
plt.plot(X, g7_rfse_minmax_3w_Y, 'k' + line_type[2] + symbol[2], linewidth=1, markersize=9, label="3W - MinMax")
plt.plot(X, g7_rfse_mix_3w_Y, 'k' + line_type[3] + symbol[3], linewidth=1, markersize=9, label="3W - Cosine & MinMax")
plt.plot(X, g7_ocsvm_3w_Y, 'k' + line_type[5], linewidth=2, markersize=9, label="3W - Baseline")

ax.annotate(
	'F1=0.782, AUC=0.843',
	xy=(0.12, 0.98), xytext=(0.2, 0.85), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'lightgray', 'alpha':0.9}
)

ax.annotate(
	'F1=0.770, AUC=0.829',
	xy=(0.08, 0.925), xytext=(0.05, 0.75), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

ax.annotate(
	'F1=0.775, AUC=0.845',
	xy=(0.83, 0.95), xytext=(0.65, 0.85), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

ax.annotate(
	'F1=0.601, AUC=0.560',
	xy=(0.8, 0.743), xytext=(0.5, 0.65), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/PR_Curves_RFSE_7Genres_3W_F1Based_11AVG.eps'



# KI04 1W for every Distance Measure
plt.plot(X, ki04_rfse_cos_1w_Y, 'k' + line_type[1] + symbol[1], linewidth=1, markersize=9, label="1W - Cosine")
plt.plot(X, ki04_rfse_minmax_1w_Y, 'k' + line_type[0] + symbol[0], linewidth=1, markersize=9, label="1W - MinMax")
plt.plot(X, ki04_rfse_mix_1w_Y, 'k' + line_type[3] + symbol[3], linewidth=1, markersize=9, label="1W - Cosine & MinMax")
plt.plot(X, ki04_ocsvm_1w_Y, 'k' + line_type[5], linewidth=2, markersize=9, label="1W - Baseline")

ax.annotate(
	'F1=0.609, AUC=0.528',
	xy=(0.85, 0.72), xytext=(0.7, 0.8), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

ax.annotate(
	'F1=0.612, AUC=0.545',
	xy=(0.74, 0.67), xytext=(0.3, 0.61), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'lightgray', 'alpha':0.9}
)

ax.annotate(
	'F1=0.605, AUC=0.516',
	xy=(0.37, 0.88), xytext=(0.15, 0.78), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

ax.annotate(
	'F1=0.395, AUC=0.141',
	xy=(0.4, 0.4), xytext=(0.25, 0.5), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/PR_Curves_RFSE_KI04_1W_F1Based_11AVG.eps'


# SANTINIS 1W for every Distance Measure
plt.plot(X, snt_rfse_cos_3w_Y, 'k' + line_type[1] + symbol[1], linewidth=1, markersize=9, label="3W - Cosine")
plt.plot(X, snt_rfse_minmax_3w_Y, 'k' + line_type[0] + symbol[0], linewidth=1, markersize=9, label="3W - MinMax")
plt.plot(X, snt_rfse_mix_3w_Y, 'k' + line_type[3] + symbol[3], linewidth=1, markersize=9, label="3W - Cosine & MinMax")
plt.plot(X, snt_ocsvm_3w_Y, 'k' + line_type[5], linewidth=2, markersize=9, label="3W - Baseline")

ax.annotate(
	'F1=0.758, AUC=0.775',
	xy=(0.17, 0.85), xytext=(0.1, 0.7), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

ax.annotate(
	'F1=0.787, AUC=0.861',
	xy=(0.45, 0.92), xytext=(0.45, 0.72), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'lightgray', 'alpha':0.9}
)

ax.annotate(
	'F1=0.768, AUC=0.876',
	xy=(0.55, 0.95), xytext=(0.69, 0.80), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

ax.annotate(
	'F1=0.395, AUC=0.141',
	xy=(0.35, 0.38), xytext=(0.15, 0.5), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/PR_Curves_RFSE_SANTINIS_3W_F1Based_11AVG.eps'



# 7Genre Distance Measure Cosine
plt.plot(X, g7_rfse_cos_3w_Y, 'k' + line_type[1] + symbol[1], linewidth=1, markersize=9, label="3W - Cosine")
plt.plot(X, g7_rfse_cos_w_Y, 'k' + line_type[0] + symbol[0], linewidth=1, markersize=9, label="1W - Cosine")
plt.plot(X, g7_rfse_cos_4c_Y, 'k' + line_type[3] + symbol[3], linewidth=1, markersize=9, label="4C - Cosine")

ax.annotate(
	'F1=0.782, AUC=0.843',
	xy=(0.65, 0.985), xytext=(0.39, 0.9), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'lightgray', 'alpha':0.9}
)

ax.annotate(
	'F1=0.733, AUC=0.771',
	xy=(0.15, 0.975), xytext=(0.05, 0.88), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

ax.annotate(
	'F1=0.755, AUC=0.808',
	xy=(0.85, 0.9), xytext=(0.65, 0.80), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/PR_Curves_RFSE_7Genres_Cosine_F1Based_11AVG.eps'

# KI04 Distance Measure MinMax
plt.plot(X, ki04_rfse_minmax_3w_Y, 'k' + line_type[1] + symbol[1], linewidth=1, markersize=9, label="3W - MinMax")
plt.plot(X, ki04_rfse_minmax_1w_Y, 'k' + line_type[0] + symbol[0], linewidth=1, markersize=9, label="1W - MinMax")
plt.plot(X, ki04_rfse_minmax_4c_Y, 'k' + line_type[3] + symbol[3], linewidth=1, markersize=9, label="4C - MinMax")

ax.annotate(
	'F1=0.589, AUC=0.491',
	xy=(0.54, 0.75), xytext=(0.3, 0.61), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

ax.annotate(
	'F1=0.612, AUC=0.545',
	xy=(0.55, 0.80), xytext=(0.55, 0.88), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'lightgray', 'alpha':0.9}
)

ax.annotate(
	'F1=0.595, AUC=0.488',
	xy=(0.35, 0.86), xytext=(0.15, 0.78), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
	
)

fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/PR_Curves_RFSE_KI04_MinMax_F1Based_11AVG.eps'



# SANTINIS Distance Measure MinMax
plt.plot(X, snt_rfse_minmax_3w_Y, 'k' + line_type[1] + symbol[1], linewidth=1, markersize=9, label="3W - MinMax")
plt.plot(X, snt_rfse_minmax_1w_Y, 'k' + line_type[0] + symbol[0], linewidth=1, markersize=9, label="1W - MinMax")
plt.plot(X, snt_rfse_minmax_4c_Y, 'k' + line_type[3] + symbol[3], linewidth=1, markersize=9, label="4C - MinMax")

ax.annotate(
	'F1=0.787, AUC=0.861',
	xy=(0.35, 0.925), xytext=(0.5, 0.96), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'lightgray', 'alpha':0.9}
)

ax.annotate(
	'F1=0.752, AUC=0.782',
	xy=(0.25, 0.825), xytext=(0.1, 0.89), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

ax.annotate(
	'F1=0.762, AUC=0.768',
	xy=(0.37, 0.845), xytext=(0.30, 0.78), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/PR_Curves_RFSE_SANTINIS_MinMax_F1Based_11AVG.eps'



# 7Genre F1 VS F05
g7_rfse_cos_3w_F05_Y = [1.0, 0.979231715202, 0.972763597965, 0.979211032391, 0.983812153339, 0.985893011093, 0.98713684082, 0.98206615448, 0.906332433224, 0.906332433224, 0.790000021458]
g7_ocsvm_3w_F05_Y = [1.0, 1.0, 1.0, 0.988041698933, 0.955814301968, 0.918949604034, 0.734381318092, 0.734381318092, 0.734381318092, 0.734381318092, 0.5671428442]
g7_rfse_cos_1w_P_Y = [1.0, 0.997481405735, 0.979513645172, 0.982258379459, 0.647446990013, 0.647446990013, 0.647446990013, 0.647446990013, 0.647446990013, 0.647446990013, 0.389285713434]
g7_ocsvm_3w_P_Y = [1.0, 1.0, 0.999636828899, 0.98711502552, 0.636886656284, 0.636886656284, 0.636886656284, 0.636886656284, 0.636886656284, 0.636886656284, 0.382142871618]
g7_ocsvm_3w_AUC_Y = [1.0, 0.974085211754, 0.967706739902, 0.966729879379, 0.973892211914, 0.978022575378, 0.979698956013, 0.979603230953, 0.967948377132, 0.92431551218, 0.8842856884]

plt.plot(X, g7_rfse_cos_3w_Y, 'k' + line_type[0] + symbol[0], linewidth=1, markersize=9, label="3W - Cos - F1")
plt.plot(X, g7_rfse_cos_3w_F05_Y, 'k' + line_type[1] + symbol[1], linewidth=1, markersize=9, label="3W - Cos - F0.5")
plt.plot(X, g7_ocsvm_3w_AUC_Y, 'k' + line_type[2] + symbol[3], linewidth=1, markersize=9, label="3W - Cos - AUC")
plt.plot(X, g7_ocsvm_4c_Y, 'k' + line_type[3] + symbol[5], linewidth=1, markersize=9, label="4C - OCSVME - F1")
plt.plot(X, g7_ocsvm_3w_F05_Y, 'k' + line_type[4] + symbol[4], linewidth=1, markersize=9, label="3W - OCSVME - F0.5")
#plt.plot(X, g7_ocsvm_3w_P_Y, 'k' + line_type[4] + symbol[4], linewidth=1, markersize=9, label="3W - OCSVME - mP")


ax.annotate(
	'F1=0.782, AUC=0.843',
	xy=(0.98, 0.88), xytext=(0.65, 0.77), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

ax.annotate(
	'F0.5=0.808, AUC=0.774',
	xy=(0.75, 0.95), xytext=(0.57, 0.85), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

ax.annotate(
	'AUC=0.857, F1=0.779',
	xy=(0.95, 0.90), xytext=(0.7, 1.03), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

ax.annotate(
	'F1=0.626, AUC=0.524',
	xy=(0.45, 0.765), xytext=(0.3, 0.68), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

ax.annotate(
	'F0.5=0.680, AUC=0.545',
	xy=(0.25, 0.99), xytext=(0.2, 1.05), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/PRC_7Genres_F1vsF05_11AVG.eps'


# KI04 F1 VS F05
ki04_rfse_cosine_F05_1w_Y = [1.0, 0.98644721508, 0.964075922966, 0.9270581007, 0.905764818192, 0.8824390769, 0.651164531708, 0.651164531708, 0.651164531708, 0.651164531708, 0.500414967537]
ki04_ocsvm_1w_F05_Y = [1.0, 0.360364258289, 0.430741429329, 0.422313332558, 0.394014149904, 0.394014149904, 0.394014149904, 0.394014149904, 0.394014149904, 0.394014149904, 0.350207477808]
ki04_ocsvm_3w_P_Y = []
ki04_rfse_cosine_P_4c_Y = []

plt.plot(X, ki04_rfse_minmax_1w_Y, 'k' + line_type[0] + symbol[0], linewidth=1, markersize=9, label="1W - MinMax - F1")
plt.plot(X, ki04_rfse_cosine_F05_1w_Y, 'k' + line_type[1] + symbol[1], linewidth=1, markersize=9, label="1W - Cos - F0.5")
plt.plot(X, ki04_rfse_cosine_P_4c_Y, 'k' + line_type[2] + symbol[2], linewidth=1, markersize=9, label="4C - Cos - F0.5")
plt.plot(X, ki04_ocsvm_1w_Y, 'k' + line_type[3] + symbol[3], linewidth=1, markersize=9, label="1W - OCSVME - F1")
plt.plot(X, ki04_ocsvm_1w_F05_Y, 'k' + line_type[4] + symbol[4], linewidth=1, markersize=9, label="1W - OCSVME - F0.5")
plt.plot(X, ki04_ocsvm_3w_P_Y, 'k' + line_type[5] + symbol[5], linewidth=1, markersize=9, label="3W - OCSVME - P")


ax.annotate(
	'F1=0.612, AUC=0.545',
	xy=(0.55, 0.80), xytext=(0.55, 0.88), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

ax.annotate(
	'F0.5=0.697, AUC=0.467',
	xy=(0.65, 0.65), xytext=(0.45, 0.50), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

ax.annotate(
	'F1=0.395, AUC=0.141',
	xy=(0.14, 0.35), xytext=(0.25, 0.3), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

ax.annotate(
	'F0.5=0.486, AUC=0.143',
	xy=(0.15, 0.4), xytext=(0.15, 0.6), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/PRC_KI04_F1vsF05_11AVG.eps'

"""

# SANTINIS Distance Measure MinMax

snt_rfse_minmax_F05_3w_Y = [1.0, 0.981221735477, 0.958775579929, 0.930858135223, 0.918592333794, 0.914032936096, 0.914665579796, 0.922808587551, 0.92455971241, 0.92622667551, 0.926093280315]
snt_ocsvm_F05_3w_Y = [1.0, 0.47872826457, 0.448499083519, 0.569252550602, 0.638963222504, 0.683330178261, 0.714452266693, 0.74061280489, 0.757108986378, 0.757108986378, 0.76315331459]
snt_rfse_cos_P_1w_Y = [1.0, 0.856296777725, 0.85298371315, 0.851261734962, 0.864491224289, 0.868298828602, 0.877465426922, 0.893401265144, 0.896014034748, 0.892807722092, 0.890418112278]
snt_ocsvm_P_3w_Y = [1.0, 0.695052742958, 0.800762414932, 0.8338509202, 0.853602766991, 0.862537264824, 0.865321874619, 0.865592479706, 0.865617752075, 0.86842495203, 0.870121955872]
snt_rfse_mix_AUC_Y = [1.0, 0.982258021832, 0.969178617001, 0.960849702358, 0.955416321754, 0.952583909035, 0.946125328541, 0.937441468239, 0.929030895233, 0.924789190292, 0.922538280487]

#snt_rfse_cos_P_1w_Y = [1.0, 1.0, 0.467419654131, 0.467419654131, 0.467419654131, 0.467419654131, 0.467419654131, 0.467419654131, 0.467419654131, 0.467419654131, 0.199285715818]
#snt_ocsvm_P_3w_Y = [1.0, 1.0, 1.0, 0.988041698933, 0.955814301968, 0.918949604034, 0.734381318092, 0.734381318092, 0.734381318092, 0.734381318092, 0.5671428442]


plt.plot(X, snt_rfse_minmax_3w_Y, 'k' + line_type[1] + symbol[1], linewidth=1, markersize=9, label="3W - MinMax - F1")
plt.plot(X, snt_rfse_minmax_F05_3w_Y, 'k' + line_type[0] + symbol[0], linewidth=1, markersize=9, label="3W - MinMax - F0.5")
plt.plot(X, snt_rfse_cos_P_1w_Y, 'k' + line_type[2] + symbol[2], linewidth=1, markersize=9, label="1W - Cosine - mP")
plt.plot(X, snt_rfse_mix_AUC_Y, 'k' + line_type[2] + symbol[3], linewidth=1, markersize=9, label="3W - C&M - mP")
plt.plot(X, snt_ocsvm_3w_Y, 'k' + line_type[5] + symbol[5], linewidth=1, markersize=9, label="3W - OCSVME - F1")
#plt.plot(X, snt_ocsvm_F05_3w_Y, 'k' + line_type[4] + symbol[4], linewidth=1, markersize=9, label="3W - OCSVME - F0.5")
plt.plot(X, snt_ocsvm_P_3w_Y, 'k' + line_type[4] + symbol[4], linewidth=1, markersize=9, label="3W - OCSVME - mP")

ax.annotate(
	'F1=0.787, F0.5=0.868, AUC=0.861',
	xy=(0.45, 0.92), xytext=(0.4, 0.73), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.4', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'lightgray', 'alpha':0.9}
)

ax.annotate(
	'F1=0.643, AUC=0.197',
	xy=(0.35, 0.38), xytext=(0.25, 0.20), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)


ax.annotate(
	'AUC=0.876, mP=0.878, F1=0.768',
	xy=(0.55, 0.95), xytext=(0.35, 1.03), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)


ax.annotate(
	'mP=0.954, AUC=0.777',
	xy=(0.25, 0.855), xytext=(0.22, 0.62), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

ax.annotate(
	'mP=0.743, AUC=0.726',
	xy=(0.15, 0.75), xytext=(0.10, 0.50), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'lightgray', 'alpha':0.9}
)

fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/PRC_SANTINIS_F1vsF05_11AVG.eps'

"""

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



#=================================================================================================
# OCSVME
#=================================================================================================

#7Genres
plt.plot(X, g7_ocsvm_3w_Y, 'k' + line_type[0] + symbol[0], linewidth=1, markersize=9, label="3W - 7Genres")
plt.plot(X, g7_ocsvm_1w_Y, 'k' + line_type[1] + symbol[1], linewidth=1, markersize=9, label="1W - 7Genres")
plt.plot(X, g7_ocsvm_4c_Y, 'k' + line_type[2] + symbol[2], linewidth=1, markersize=9, label="4C - 7Genres")

ax.annotate(
	'F1=0.601, AUC=0.560',
	xy=(0.55, 0.82), xytext=(0.6, 0.9), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

ax.annotate(
	'F1=0.582, AUC=0.454',
	xy=(0.63, 0.63), xytext=(0.7, 0.53), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

ax.annotate(
	'F1=0.626, AUC=0.524',
	xy=(0.46, 0.76), xytext=(0.2, 0.6), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'lightgray', 'alpha':0.9}
)

fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/PR_Curves_OCSVM_7Genres_F1Based_11AVG.eps'



#KI04
plt.plot(X, ki04_ocsvm_3w_Y, 'k' + line_type[0] + symbol[0], linewidth=1, markersize=9, label="3W - KI04")
plt.plot(X, ki04_ocsvm_1w_Y, 'k' + line_type[1] + symbol[1], linewidth=1, markersize=9, label="1W - KI04")
plt.plot(X, ki04_ocsvm_4c_Y, 'k' + line_type[3] + symbol[3], linewidth=1, markersize=9, label="4C - KI04")

ax.annotate(
	'F1=0.342, AUC=0.173',
	xy=(0.25, 0.53), xytext=(0.3, 0.63), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

ax.annotate(
	'F1=0.395, AUC=0.141',
	xy=(0.13, 0.34), xytext=(0.1, 0.15), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'lightgray', 'alpha':0.9}
)

ax.annotate(
	'F1=0.388, AUC=0.153',
	xy=(0.14, 0.425), xytext=(0.25, 0.3), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/PR_Curves_OCSVM_KI04_F1Based_11AVG.eps'



#SANTINIS
plt.plot(X, snt_ocsvm_3w_Y, 'k' + line_type[6] + symbol[0], linewidth=1, markersize=9, label="3W - SANTINIS")
plt.plot(X, snt_ocsvm_1w_Y, 'k' + line_type[7] + symbol[1], linewidth=1, markersize=9, label="1W - SANTINIS")
plt.plot(X, snt_ocsvm_4c_Y, 'k' + line_type[8] + symbol[3], linewidth=1, markersize=9, label="4C - SANTINIS")

ax.annotate(
	'F1=0.643, AUC=0.197',
	xy=(0.44, 0.44), xytext=(0.55, 0.35), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'lightgray', 'alpha':0.9}
)

ax.annotate(
	'F1=0.270, AUC=0.038',
	xy=(0.28, 0.2), xytext=(0.35, 0.1), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=-0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

ax.annotate(
	'F1=0.472, AUC=0.222',
	xy=(0.46, 0.5), xytext=(0.3, 0.63), fontsize=16,
	arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3,rad=0.3', 'facecolor':'black'},
	bbox={'boxstyle':'round,pad=0.5','facecolor':'white', 'alpha':0.9}
)

fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/PR_Curves_OCSVM_SANTINIS_F1Based_11AVG.eps'

"""

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



#=================================================================================================
# Plot it!
#=================================================================================================

plt.grid(True)
#plt.legend(loc='upper left', bbox_to_anchor=(0.62, 0.4), ncol=1, fancybox=True, shadow=True, fontsize=16)
plt.legend(loc=4, fancybox=True, shadow=True, fontsize=16)
plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1], fontsize=16)
plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=16)
plt.tight_layout()

plt.savefig(fig_save_file, bbox_inches='tight')

plt.show()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++