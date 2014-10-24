
import matplotlib.pyplot as plt

X = [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 ]

#7-Genres
g7_ocsvm_Y = [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9976089, 0.98668551, 0.98362058, 0.97953314, 0.62914503 ] 
g7_rfse_minmax_3w_Y = [ 1.0, 1.0, 0.98827606, 0.97182566, 0.97624469, 0.98017693, 0.98087931, 0.98242229, 0.98420459, 0.9832086, 0.84219867 ]
g7_rfse_minmax_1w_Y = [ 1.0, 0.95884335, 0.97442234, 0.97153389, 0.97197568, 0.97287112, 0.96503079, 0.96171665, 0.95216388, 0.93052506, 0.68214393 ]
g7_rfse_minmax_4c_Y = [ 1.0, 0.99126208, 0.99051529, 0.98656279, 0.98738045, 0.98782933, 0.98531514, 0.98534298, 0.98655701, 0.98371291, 0.7266348 ]
g7_rfse_cos_3w_Y = [ 1.0, 0.98864573, 0.98097557, 0.97369921, 0.97656375, 0.97761798, 0.97989279, 0.97790807, 0.97761202, 0.97253358, 0.92674059 ] 
g7_rfse_cos_1w_Y = [ 1.0, 1.0, 0.99977893, 0.99574125, 0.99493581, 0.9953239, 0.9952662, 0.98928064, 0.98089439, 0.97269535, 0.83531505 ]
g7_rfse_cos_4c_Y = [ 0.5, 0.65855801, 0.71602899, 0.68922704, 0.69640326, 0.67582589, 0.68556321, 0.69222242, 0.69527912, 0.70503098, 0.45355988 ]
g7_fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/ECIR2015/diagrams/PR_Curves_7Genres_MinMax_Baseline.eps'


#SANTINIS
snt_ocsvm_Y = [ 1.0, 0.6854369, 0.79070109, 0.82635659, 0.84602731, 0.85820764, 0.86403847, 0.86529154, 0.86567241, 0.86537236, 0.86811781 ] 
snt_rfse_cos_3w_Y = [ 1.0, 0.94446266, 0.82702887, 0.79409623, 0.82106698, 0.8442868, 0.85818356, 0.87251091, 0.88569158, 0.89047933, 0.89260876] 
snt_rfse_cos_1w_Y = [ 1.0, 0.82726043, 0.84945041, 0.87669706, 0.88373548, 0.88201118, 0.89774966, 0.90478861, 0.90936548, 0.89992571, 0.90156853]
snt_rfse_cos_4c_Y = [ 1.0, 0.8335436, 0.86719489, 0.88362503, 0.88739169, 0.88533372, 0.9004938, 0.90726352, 0.91165179, 0.90215689, 0.90329415] 
snt_rfse_minmax_3w_Y = [ 1.0, 0.97936189, 0.9034133, 0.86187643, 0.88911998, 0.8908453, 0.90072566, 0.90865272, 0.91795838, 0.92119819, 0.91894877]
snt_rfse_minmax_1w_Y = [1.0, 0.77683818, 0.78050381, 0.83782291, 0.85853809, 0.86635935, 0.88349831, 0.89586574, 0.90412492, 0.90252841, 0.90403175] 
snt_rfse_minmax_4c_Y = [ 1.0, 0.74207705, 0.76509899, 0.82534927, 0.84927046, 0.85951352, 0.87775457, 0.89100772, 0.89967281, 0.89879537, 0.89992547]
snt_fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/ECIR2015/diagrams/PR_Curves_SANTINIS_MinMax&Cosine_Baseline.eps'

#KI04
ki04_ocsvm_Y = [ 1.0, 0.92144424, 0.86137992, 0.87606782, 0.82173979, 0.80229491, 0.81115234, 0.80875248, 0.78558546, 0.78004068, 0.2980237 ] 
ki04_rfse_cos_3w_Y = [ 1.0, 1.0, 0.98869789, 0.95833653, 0.9409101, 0.93616289, 0.93559998, 0.93137866, 0.92188543, 0.90691775, 0.57851332]
ki04_rfse_cos_1w_Y = [ 1.0, 1.0, 0.98629916, 0.97612298, 0.96284038, 0.95470977, 0.95565826, 0.93749541, 0.92429531, 0.91704082, 0.6058346 ] 
ki04_rfse_cos_4c_Y = [ 1.0, 1.0, 1.0, 0.98993838, 0.96718925, 0.96011651, 0.95810366, 0.9609344, 0.95956022, 0.95561016, 0.43619961]
ki04_rfse_minmax_3w_Y = [ 1.0, 0.99438697, 0.93540812, 0.91854012, 0.92372507, 0.91133505, 0.92397946, 0.92647862, 0.92626494, 0.91577607, 0.43672138 ]
ki04_rfse_minmax_1w_Y = [ 1.0, 1.0, 1.0, 0.99868935, 0.98461539, 0.9755165, 0.96798366, 0.96075344, 0.94272643, 0.93637425, 0.47655249]
ki04_rfse_minmax_4c_Y = [ 1.0, 0.98973924, 0.9290722, 0.90901953, 0.90708631, 0.90478307, 0.89457405, 0.86959368, 0.86093277, 0.86178279, 0.55028629 ]
ki04_fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/ECIR2015/diagrams/PR_Curves_KI04_Cosine_Baseline.eps'

symbol = [ 'o', 'v', '^', '*', '<', 's', '+', 'x', '>', 'H', '1', '2', '3', '4', 'D', 'h', '8', 'd', 'p', '.', ',' ]
line_type = [ '--', '--', '--', '-', '-', '-', '-', '-', '-', '--', '--', '--', '--.', '-' , '-', '-', '-', '--', '--', '--', '--']

plt.figure(num=1, figsize=(12, 7), dpi=80, facecolor='w', edgecolor='k')

#plt.plot(X, g7_ocsvm_Y, 'k' + line_type[5], linewidth=2, markeredgewidth=1, label="Baseline")  
#plt.plot(X, g7_rfse_minmax_3w_Y, 'k' + line_type[0] + symbol[0], linewidth=1, markersize=9, label="3W - MinMax")
#plt.plot(X, g7_rfse_minmax_1w_Y, 'k' + line_type[1] + symbol[1], linewidth=1, markersize=9, label="1W - MinMax") 
#plt.plot(X, g7_rfse_minmax_4c_Y, 'k' + line_type[2] + symbol[2], linewidth=1, markersize=9, label="4C - MinMax")
#plt.plot(X, g7_rfse_cos_3w_Y, 'k' + line_type[3] + symbol[3], linewidth=1, markersize=9, label="3W - Cosine")  
#plt.plot(X, g7_rfse_cos_1w_Y, 'k' + line_type[4] + symbol[4], linewidth=1, markersize=9, label="1W - Cosine")    
#plt.plot(X, g7_rfse_cos_4c_Y, 'k' + line_type[5] + symbol[5], linewidth=1, markersize=9, label="4C - Cosine")  
##
#plt.plot(X, ki04_ocsvm_Y, 'k' + line_type[5], linewidth=2, markersize=9, label="Baseline")  
#plt.plot(X, ki04_rfse_cos_3w_Y, 'k' + line_type[0] + symbol[0], linewidth=1, markersize=9, label="3W - Cosine")
#plt.plot(X, ki04_rfse_cos_1w_Y, 'k' + line_type[1] + symbol[1], linewidth=1, markersize=9, label="1W - Cosine") 
#plt.plot(X, ki04_rfse_cos_4c_Y, 'k' + line_type[2] + symbol[2], linewidth=1, markersize=9, label="4C - Cosine") 
#plt.plot(X, ki04_rfse_minmax_3w_Y, 'k' + line_type[3] + symbol[3], linewidth=1, markersize=9, label="3W - MinMax")   
#plt.plot(X, ki04_rfse_minmax_1w_Y, 'k' + line_type[4] + symbol[4], linewidth=1, markersize=9, label="1W - MinMax")   
#plt.plot(X, ki04_rfse_minmax_4c_Y, 'k' + line_type[5] + symbol[5], linewidth=1, markersize=9, label="4C - MinMax")   
#

plt.plot(X, snt_ocsvm_Y, 'k' + line_type[5], linewidth=2, markersize=9, label="Baseline")  
plt.plot(X, snt_rfse_cos_3w_Y, 'k' + line_type[0] + symbol[0], linewidth=1, markersize=9, label="3W - Cosine")
plt.plot(X, snt_rfse_cos_1w_Y, 'k' + line_type[1] + symbol[1], linewidth=1, markersize=9, label="1W - Cosine") 
plt.plot(X, snt_rfse_cos_4c_Y, 'k' + line_type[2] + symbol[2], linewidth=1, markersize=9, label="4C - Cosine")  
plt.plot(X, snt_rfse_minmax_3w_Y, 'k' + line_type[3] + symbol[3], linewidth=1, markersize=9, label="3W - MinMax")
plt.plot(X, snt_rfse_minmax_1w_Y, 'k' + line_type[4] + symbol[4], linewidth=1, markersize=9, label="1W - MinMax") 
plt.plot(X, snt_rfse_minmax_4c_Y, 'k' + line_type[5] + symbol[5], linewidth=1, markersize=9, label="4C - MinMax")  

plt.grid(True)
plt.legend(loc=8, fancybox=True, shadow=True)
plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.02])
plt.tight_layout()

plt.savefig(g7_fig_save_file, bbox_inches='tight')

plt.show()
