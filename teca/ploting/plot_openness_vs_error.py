
import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

fig_save_file = '/home/dimitrios/Documents/MyPublications:Journals-Conferences/Journal_IPM-Elsevier/diagrams/MacroPRC11AVG_RFSE_OCSVME_SANTINIS.eps'

# RFSE MIX W3G
y1 = np.array([
    0.49214719046625,
    0.442032251123,
    0.41277414618375,
    0.42694523675275,
    0.42946733236725,
    0.45145857728525,
    0.5160593162845
])

y1_error = np.array([
    0.053294279988581,
    0.039755900665204,
    0.008452923400279,
    0.024859871589693,
    0.051999834685487,
    0.025323468112743,
    0.011765826733599
])

# RFSE MIX C4G
y2 = np.array([
    0.6272131808775,
    0.59434220948575,
    0.52331215942875,
    0.4626093081945,
    0.44846815257225,
    0.4605380358385,
    0.02918786963565
])

y2_error = np.array([
    0.028287307251918,
    0.041048841914448,
    0.03571369345865,
    0.040649475976874,
    0.024158064398117,
    0.011384673083569,
    0.002806890332018
])

# RFSE MIX W1G
y3 = np.array([
    0.5567991616475,
    0.568474887288,
    0.509841539021,
    0.4556869523045,
    0.419280236414,
    0.44529397659325,
    0.267823366977025
])

y3_error = np.array([
    0.004150950875243,
    0.062716195475856,
    0.042222962248249,
    0.067439180994244,
    0.026148491738274,
    0.007292432068356,
    0.008120568211084
])

# RFSE MinMax W3G
y4 = np.array([
    0.3956261259115,
    0.389109852237,
    0.34863422534725,
    0.405019241471,
    0.45848410243525,
    0.4625533195155,
    0.583494314843
])

y4_error = np.array([
    0.063269655055877,
    0.057190742234244,
    0.04051203965903,
    0.061541586776941,
    0.096079819720949,
    0.022749332177277,
    0.03513837280362
])

# RFSE MinMax C4G
y5 = np.array([
    0.6089139251405,
    0.6007454396,
    0.55466646645825,
    0.52027944344475,
    0.4826792067925,
    0.496176184569,
    0.26742231486005,
])

y5_error = np.array([
    0.022146332133328,
    0.011141999971776,
    0.033922106840503,
    0.02727149034468,
    0.050973628002414,
    0.036120934968928,
    0.272741568616874
])

# RFSE MinMax W1G
y6 = np.array([
    0.42457465406,
    0.462851750479,
    0.452765797578,
    0.48985706057125,
    0.413050376716,
    0.4767259530135,
    0.38141030316975
])

y6_error = np.array([
    0.044591405111714,
    0.07813845772564,
    0.026912424567281,
    0.066871753637413,
    0.016114299009104,
    0.01752979777425,
    0.246555758723686
])

# OCSVME 0.1 W3G
y7 = np.array([
    0.40212930152375,
    0.35782352221425,
    0.3510319454015,
    0.33928030086625,
    0.376352369348,
    0.44351647267975,
    0.562976457851
])

y7_error = np.array([
    0.030057388871947,
    0.025888846335394,
    0.015590413686572,
    0.025131625342932,
    0.03051068827693,
    0.021829609875934,
    0.032478970179
])

# OCSVME 0.1 C4G
y8 = np.array([
    0.43360580244375,
    0.42952878883875,
    0.418083771738,
    0.399930816164,
    0.42402834243475,
    0.46200077710225,
    0.52294539451425
])

y8_error = np.array([
    0.026625895683506,
    0.027124286310632,
    0.020084728019197,
    0.040338742566734,
    0.032094542894286,
    0.053337170392383,
    0.039283975697979
])


# OCSVME 0.1 W1G
y9 = np.array([
    0.37444311583625,
    0.36965519779725,
    0.38495172108225,
    0.37371188148725,
    0.37078617625175,
    0.4353245703605,
    0.527513452058
])

y9_error = np.array([
    0.050911321711413,
    0.049023273382749,
    0.043127593002661,
    0.054186146404814,
    0.035578989674327,
    0.015682852802,
    0.03208954528538
])

# OCSVME 0.07 W3G
y10 = np.array([
    0.38262572549325,
    0.3413473300765,
    0.338381784272,
    0.33579715064875,
    0.37284905192825,
    0.44260539303975,
    0.562467309426
])

y10_error = np.array([
    0.031125532965101,
    0.031054378528259,
    0.015176906418823,
    0.01956680272715,
    0.027295755593335,
    0.020205668550464,
    0.03349826106257
])

# OCSVME 0.07 C4G
y11 = np.array([
    0.41833816872325,
    0.41718451560025,
    0.3962246891665,
    0.38414248993625,
    0.41796562477425,
    0.46461628530025,
    0.52434856615225
])

y11_error = np.array([
    0.021643548463433,
    0.025248758854881,
    0.015656779180852,
    0.03980402077346,
    0.036421429896241,
    0.046183853738898,
    0.03827802333588
])

# OCSVME 0.07 W1G
y12 = np.array([
    0.37350040666175,
    0.36485774539375,
    0.37973694922525,
    0.37239762346225,
    0.36861425284675,
    0.43569745524875,
    0.5297022689055
])

y12_error = np.array([
    0.052144433435577,
    0.046781665372574,
    0.03197742972219,
    0.046644528492772,
    0.035237925931609,
    0.014286682400299,
    0.03045216556161
])

x_openness_leves = np.array([
    0.069,
    0.152,
    0.251,
    0.368,
    0.503,
    0.657,
    0.825
])

yp1 = np.array([
    0.644895091198,
    0.548335335367,
    0.48169356359375,
    0.465786645681,
    0.480671542584,
    0.4089122744175,
    0.5105169872985
])

yp1_error = np.array([
    0.082595076205826,
    0.073187219221394,
    0.054416376391735,
    0.046342324484229,
    0.111316366995677,
    0.020864127946367,
    0.003529184151526
])

yp2 = np.array([
    0.6494414306535,
    0.59576231052525,
    0.52415687927775,
    0.44941315381575,
    0.380000150043,
    0.42531040327375,
    0.26154860002615
])

yp2_error = np.array([
    0.025836313767528,
    0.037204175374653,
    0.073669607716422,
    0.058190673585438,
    0.06612150745628,
    0.110812947008654,
    0.24
])

yp3 = np.array([
    0.76790426440175,
    0.67857091536525,
    0.52044941112825,
    0.52372047617875,
    0.37359423934325,
    0.3800907948675,
    0.382041190439525
])

yp3_error = np.array([
    0.023189970053264,
    0.079773753356009,
    0.070873245412173,
    0.054211084555288,
    0.036813744417413,
    0.026557242917503,
    0.250794764774296
])

ypBL = np.array([
    0.534467228811,
    0.440739396065,
    0.382135065752,
    0.33693243117525,
    0.3409672651495,
    0.362357236698,
    0.504659140762
])

ypBL_error = np.array([
    0.020464082145689,
    0.049761542144249,
    0.02397097876023,
    0.02601068723454,
    0.053212177080931,
    0.014366111923426,
    0.001712190798643
])

# x_openness_leves = np.array([
#     0.03,
#     0.04,
#     0.05,
#     0.06,
#     0.07,
#     0.11,
#     0.18
# ])


fig = plt.figure(num=1, figsize=(12, 7), facecolor='w', edgecolor='k')  # dpi=300,
ax = fig.add_subplot(111)

linestyle = {
    "linestyle": "-",
    "marker": "o",
    "linewidth": 2,
    "markeredgewidth": 2,
    'markeredgecolor': 'white',
    "elinewidth": 1,
    "capsize": 4
}


linestyle['color'] = 'orange'
ax.errorbar(x_openness_leves, yp1, yerr=yp1_error, **linestyle)

linestyle['color'] = 'purple'
ax.errorbar(x_openness_leves, yp2, yerr=yp2_error, **linestyle)

linestyle['color'] = 'green'
ax.errorbar(x_openness_leves, yp3, yerr=yp3_error, **linestyle)

linestyle['color'] = 'black'
linestyle['linestyle'] = "--"
ax.errorbar(x_openness_leves, ypBL, yerr=ypBL_error, **linestyle)

# linestyle['color'] = 'lime'
# ax.errorbar(x_openness_leves, y11, yerr=y11_error, **linestyle)

# linestyle['color'] = 'purple'
# ax.errorbar(x_openness_leves, y12, yerr=y12_error, **linestyle)

ax.yaxis.grid()

ln1 = mlines.Line2D([], [], markersize=0, linewidth=3, color='orange')
ln2 = mlines.Line2D([], [], markersize=0, linewidth=3, color='purple')
ln3 = mlines.Line2D([], [], markersize=0, linewidth=3, color='green')
ln4 = mlines.Line2D([], [], markersize=0, linewidth=3, color='black')
ln5 = mlines.Line2D([], [], markersize=0, linewidth=3, color='lime')
ln6 = mlines.Line2D([], [], markersize=0, linewidth=3, color='purple')
lndump = mlines.Line2D([], [], markersize=0, linewidth=0)

plt.legend(
    [ln1, ln2, ln3, ln4], # , ln4, ln5, ln6],
    ["W3G", "C4G", "W1G", "Baseline"],
     # "W3G - nu=0.07", "C4G - nu=0.07", "W1G - nu=0.07"],
    bbox_to_anchor=(0.0, 1.01, 1.0, 0.101),
    loc=3, ncol=4, mode="expand", borderaxespad=0.0,
    fancybox=False, shadow=False, fontsize=14
).get_frame().set_linewidth(0.0)

plt.yticks(fontsize=12)
plt.xticks(x_openness_leves, fontsize=12)
plt.ylabel('Precision', fontsize=14)
plt.xlabel('Openness Levels', fontsize=14)
# plt.tight_layout()

# Saving the ploting to File
# plt.savefig(fig_save_file, bbox_inches='tight')

plt.show()
