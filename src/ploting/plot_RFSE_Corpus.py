





kfolds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

params_range = {
    '1.Vocab' : [10000, 50000, 100000],
    '2.Feat' : [1000, 5000],
    '3.Bagging' : [0.66],
    '4.Iterations' : [100],
    '5.Sigma' : [0.5],
} 

res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI04_TT-Char4Grams-Koppels-Bagging_method_kfolds-10_GridSearch_TEST.h5', 'r')


color = ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k' ]
symbol = [ "^", "*", "x", "+", "*", "^", "x", "+" ]
line_type = [ "--", "--", "--", "--", "-" , "-", "-", "-" ]

plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(2, 1, 1)

Zero_Dist_lst = list()

for i, params in enumerate(grid_search.IterGrid(params_range)):

    params_path = "/".join( [ key.split('.')[1] + str(value).replace('.','') for key, value in sorted( params.items() ) ] )
    params_path = '/' + params_path
    
    X, Y, mark_X, mark_Y = prcurve(res_h5file, kfolds, params_path, genre_tag=3)

    Zero_Dist_lst.append( zero_class_dist(res_h5file, kfolds, params_path, genre_tag=3) )

    plt.plot(X, Y, color[i] + symbol[i] + line_type[i], markeredgewidth=2, label="("+str(i+1)+") RFSE"+params_path  )
    
    if mark_X !=None:
        plt.plot(mark_X, mark_Y, color[i] + symbol[i], markeredgewidth=15)

    plt.title( " ".join(params_path.split('/')) )
    plt.grid(True)    
    plt.legend(loc=3)

    
plt.subplot(2, 1, 2)
plt.boxplot(Zero_Dist_lst)

plt.show()

                                                                           

res_h5file.close()
    