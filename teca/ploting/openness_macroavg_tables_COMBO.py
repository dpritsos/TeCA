# !/usr/bin/env python

import sys
import tables as tb
import numpy as np
import collections as coll
import time as tm
import json
sys.path.append('../../teca')
sys.path.append('../../../DoGSWrapper/dogswrapper')

import tools.paramcombs as param_comb
from prconf_tables import PRConf_table
from auc_tables import params_prauc_tables


# Run for all cases
if __name__ == '__main__':

    # Parameters used for the experiments required for selecting specific or group of results
    kfolds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # FOR OPENNESS
    case_od = coll.OrderedDict([
        ('terms_type', ['POS3G']),
        ('corpus', ['KI04']),
        ('vocab_size', [43]),
        ('features_size', [10]),
        ('nu', [0.05]),
        # ('sim_func', ['combo']),  # 'cosine_sim', 'minmax_sim'
        # ('Sigma', [0.5]),
        # ('Iterations', [1000]),
        ('uknw_ctgs_num', [1, 2, 3, 4, 5, 6, 7]),
        ('uknw_ctgs_num_splt_itrs', [0, 1, 2, 3, 4, 5, 6, 7, 8]),
        ('kfolds', ['']),
    ])

    """
    # FOR OPEN-SET
    case_od = coll.OrderedDict([
        ('terms_type', ['POS3G']),
        ('corpus', ['SANTINIS']),
        ('vocab_size', [16200]),   # 5000, 10000, 50000, 100000, , 1330, 16200,
        ('features_size', [4, 10, 20, 40, 100, 500, 1000, 5000, 10000]),
        # 4, 10, 20, 40, 100, 500, 1000, 5000, 10000
        # 100, 500, 1000 , 5000, 10000, 50000, 90000
        ('sim_func', ['MIX']),  # , 'cosine_sim', 'minmax_sim'
        ('Sigma', [0.5, 0.7, 0.9]),
        ('Iterations', [200, 300, 500, 1000]),
        # ('nu', [0.05, 0.07, 0.1, 0.15, 0.17, 0.3, 0.5, 0.7, 0.9]),
        ('marked_uknw_ctg_lst', [12]),
        ('kfolds', ['']),
    ])
    """

    """
    case_od = coll.OrderedDict([
        ('doc_rep', ['C4G']),
        ('corpus', ['KI04']),  # , 'SANTINIS', 'KI04', '7Genres'
        ('dist', ['']),  # '', 'MinMax', 'MIX'
        ('vocab_size', [50000]),
        ('features_size', [10000]),
        ('nu', [0.07,  0.1]),
        # ('Sigma', [0.7]),
        # ('Iterations', [100]),
        ('onlytest_gnrs_splts', [1, 2, 3, 4, 5, 6, 7]),  # 1, 2, 3, 4, 5, 6,
        ('onlytest_splt_itrs', [0, 1, 2, 3, 4, 5, 6, 7]),  # 1, 2, 3
        ('kfolds', [''])
    ])

    case_od = coll.OrderedDict([
        ('doc_rep', ['W1G']),
        ('corpus', ['KI04']),  # , 'SANTINIS', 'KI04', '7Genres'
        ('dist', ['']),  # '', 'MinMax', 'MIX'
        ('vocab_size', [100000]),
        ('features_size', [100000]),
        ('nu', [0.05, 0.07, 0.1, 0.15, 0.17, 0.3, 0.5, 0.7, 0.9]),
        ('uknw_ctgs_num', [1]),  # [1, 2, 3, 4, 5, 6, 7]),
        ('uknw_ctgs_num_splt_itrs', [0, 1, 2, 3, 4, 5, 6, 7]),
        ('kfolds', ['']),
    ])

    case_od = coll.OrderedDict([
        ('doc_rep', ['W1G']),
        ('corpus', ['KI04']),  # , 'SANTINIS', 'KI04', '7Genres'
        ('dist', ['']),  # '', 'MinMax', 'MIX'
        ('vocab_size', [0]),
        ('features_size', [25, 50, 100]),
        ('sim_func', ['cosine_sim']),
        ('Sigma', [0.5, 0.7, 0.9]),
        ('Iterations', [10, 50, 100]),
        # ('nu', [0.05, 0.07, 0.1, 0.15, 0.17, 0.3, 0.5, 0.7, 0.9]),
        ('dims', [50, 100, 250, 500, 1000]),
        ('min_trm_fq', [3, 10]),
        ('win_size', [3, 8, 20]),
        ('algo', ['PV-DBOW']),
        ('alpha', [0.025]),
        ('min_alpha', [0.025]),
        ('epochs', [1, 3, 10]),
        ('decay', [0.002, 0.02]),
        ('uknw_ctgs_num', [1, 2, 3, 4, 5, 6, 7]),
        ('uknw_ctgs_num_splt_itrs', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
        ('kfolds', ['']),
    ])

    case_od = coll.OrderedDict([
        ('doc_rep', ['C4G']),
        ('corpus', ['KI04']),  # , 'SANTINIS', 'KI04', '7Genres'
        ('dist', ['']),  # '', 'MinMax', 'MIX'
        ('vocab_size', [0]),
        ('features_size', [25, 50, 100]),
        ('sim_func', ['cosine_sim']),
        # ('Sigma', [0.5, 0.7, 0.9]),
        # ('Iterations', [10, 50, 100]),
        ('nu', [0.05, 0.07, 0.1, 0.15, 0.17, 0.3, 0.5, 0.7, 0.9]),
        ('dims', [50, 100, 250, 500, 1000]),
        ('min_trm_fq', [3, 10]),
        ('win_size', [3, 8, 20]),
        ('algo', ['PV-DBOW']),
        ('alpha', [0.025]),
        ('min_alpha', [0.025]),
        ('epochs', [1, 3, 10]),
        ('decay', [0.002, 0.02]),
        ('uknw_ctgs_num', [1, 2, 3, 4, 5, 6, 7]),
        ('uknw_ctgs_num_splt_itrs', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
        ('kfolds', ['']),
    ])

    case_od = coll.OrderedDict([
        ('doc_rep', ['C4G']),
        ('corpus', ['KI04']),  # , 'SANTINIS', 'KI04', '7Genres'
        ('dist', ['']),  # '', 'MinMax', 'MIX'
        ('vocab_size', [500]),
        ('features_size', [0]),
        ('split_ptg', [0.7]),
        ('ukwn_slt_ptg', [0.3]),
        ('rt_lims_stp', [[0.5, 1.0, 0.2]]),
        ('lmda', [0.2, 0.5, 0.7]),
        ('dims', [50, 100, 250]),
        ('min_trm_fq', [3]),
        ('win_size', [8]),
        ('algo', ['PV-DBOW']),
        ('alpha', [0.025]),
        ('min_alpha', [0.025]),
        ('epochs', [3]),
        ('decay', [0.002]),
        ('uknw_ctgs_num', [1, 4]),
        ('uknw_ctgs_num_splt_itrs', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
        ('kfolds', ['']),
    ])
    """
    # List of all macro averaging precision recall values.
    pr_macro_lst = list()


    # Creating tables for tall the above cases.
    for case, case_path in zip(param_comb.ParamGridIter(case_od, 'list'),
                               param_comb.ParamGridIter(case_od, 'path')):

        if True:
        # if case[2] >= case[3]:
        # if case[8] >= case[4]:

            # Starting timing for this loop for monitoring the cosuption of the socre calculations.
            start_tm = tm.time()

            #  Selecting files to open and setting the mix flag on/off
            mix = False

            if case[4] == 'combo':
                # h5d_fl1 = tb.open_file(h5d_fl + '.h5', 'r')
                # h5d_fl2 = tb.open_file(h5d_fl + '_minmax.h5', 'r')
                # h5d_fl1 = tb.open_file('/home/dimitrios/Synergy-Crawler/KI-04/Openness_RFSE_COS_W1G_KI04/Openness_RFSE_COS_W1G_KI04_2016_10_19.h5', 'r')
                # h5d_fl1 = tb.open_file('/home/dimitrios/Synergy-Crawler/KI-04/Openness_RFSE_MinMax_W1G_KI04_ONLY_7/Openness_RFSE_MinMax_W1G_ONLY_7_KI04_2016_10_23.h5', 'r')
                # h5d_fl1 = tb.open_file('/media/dimitrios/TurnstoneDisk/KI-04/Openness_RFSE_COS_and_MinMax_C4G_KI04_8Iter_MaxNorm/Openness_RFSE_COS_C4G_KI04_8Iter_2016_12_04.h5', 'r')
                # h5d_fl2 = tb.open_file('/home/dimitrios/Synergy-Crawler/KI-04/Openness_RFSE_MinMax_W1G_KI04_FOR_MIX_ONLY/Openness_RFSE_MinMax_W1G_KI04_2016_10_21.h5', 'r')
                # h5d_fl2 = tb.open_file('/home/dimitrios/Synergy-Crawler/KI-04/Openness_RFSE_COS_W1G_KI04_ONLY_7/Openness_RFSE_COS_W1G_ONLY_7_KI04_2016_10_22.h5', 'r')

                h5d_fl1 = tb.open_file(
                    '/media/dimitrios/TurnstoneDisk/KI-04/Openness_POS3G_KI04/' +\
                    'Openness_RFSE_POS3G_KI04_2018_03_23.h5',
                    'r')
                h5d_fl2 = None

                mix = True

            else:
                # h5d_fl1 = tb.open_file(h5d_fl + '.h5', 'r')
                # h5d_fl1 = tb.open_file('/home/dimitrios/Synergy-Crawler/KI-04/Openness_RFSE_COS_W3G_KI04/Openness_RFSE_COS_W3G_KI04_2016_10_14.h5', 'r')
                # h5d_fl1 = tb.open_file('/home/dimitrios/Synergy-Crawler/KI-04/Openness_OCSVME_W1G_KI04/Openness_OCSVME_W1G_KI04_2016_10_19.h5', 'r')
                # h5d_fl1 = tb.open_file('/media/dimitrios/TurnstoneDisk/KI-04/Openness_OCSVME_C4G_KI04_8Iter_MaxNorm/Openness_OCSVME_C4G_KI04_8Iter_2016_02_04.h5', 'r')
                h5d_fl1 = tb.open_file(
                    '/media/dimitrios/TurnstoneDisk/KI-04/Openness_POS3G_KI04/' +\
                    'Openness_OCSVME_POS3G_KI04_2018_03_25.h5', 'r'
                )
                h5d_fl2 = None

            # ### Calculating the PR Curves ###
            # Preparing input for params_prauc_tables().
            param_od = coll.OrderedDict([
                ('terms_type', [case[0]]),
                ('vocab_size', [case[2]]),
                # ('split_ptg', [case[4]]),
                # ('ukwn_slt_ptg', [case[5]]),
                # ('rt_lims_stp', [case[6]]),
                # ('lmda', [case[7]]),
                ('features_size', [case[3]]),
                # ('nu', [case[4]]),
                # ('sim_func', [case[4]]),
                # ('Sigma', [case[5]]),  #
                # ('Iterations', [case[6]]),  #
                # ('marked_uknw_ctg_lst', [case[7]]),
                # ('dims', [case[7]]),
                # ('min_trm_fq', [case[8]]),
                # ('win_size', [case[9]]),
                # ('algo', [case[10]]),
                # ('alpha', [case[11]]),
                # ('min_alpha', [case[12]]),
                # ('epochs', [case[13]]),
                # ('decay', [case[14]]),
                # ('uknw_ctgs_num', [case[7]]),
                # ('uknw_ctgs_num_splt_itrs', [case[8]]),
                # ('onlytest_gnrs_splts', [case[6]]),
                # ('onlytest_splt_itrs', [case[7]]),
                ('kfolds', [''])
            ])

            """
            pr_aucz_var_table = params_prauc_tables(
                h5d_fl1, h5d_fl2, 'multiclass_macro', kfolds, param_od, mix, strata=None, trec=False
            )
            """


            # ### Calculating Marco Averaging of Precision, Recall, F1, F0.5 ###

            # Reformationg parametera path to be given properly to PRConf_table().
            print str(case_path.split('/')[1])
            params_path = '/' + str(case_path.split('/')[1]) + '/' +\
                '/'.join(case_path.split('/')[3::])

            # print pr_tabel_fname

            # Calculating the Precision and Recall Scores per Genre. Precision for a Genre is...
            # ...only calculated when there is at least one sample being counted for this Genre.
            pre_cls_vect, rcl_cls_vect = PRConf_table(
                h5d_fl1, h5d_fl2, kfolds, params_path, mix, strata=None, prereccon=1
            )

            # NOTE: The Precision and Recall vectors of scores per Genre might not have the same...
            # ...leght due to Unknown_Class tag expected or not expected case or just because...
            # ...for some Class we have NO Predicitons at all.
            macro_p = np.mean(pre_cls_vect)
            macro_r = np.mean(rcl_cls_vect)

            f1 = 2.0 * macro_p*macro_r / (macro_p+macro_r)
            if np.isnan(f1):
                f1 = 0

            # f05 = 1.25 * macro_p*macro_r / (0.25*macro_p+macro_r)
            # if np.isnan(f05):
            #     f05 = 0

            # pr_mean = (macro_pr[0]+macro_pr[1]) / 2.0
            # print pr_aucz_var_table

            # pr_auc = pr_aucz_var_table[0, 8]  # For RFSE is 8, for OCSVME 6, ECCE 5

            # maucs_num = len(m_aucz_var_table[0, 4::])  # For RFSE is 4, for OCSVME 3
            # m_auc_f1 = maucs_num / np.sum([1.0/mauc for mauc in m_aucz_var_table[0, 3::]])
            # if np.isnan(m_auc_f1):
            #     m_auc_f1 = 0

            # Macro Averaging AUCs per Genre.
            # case.extend([macro_p, macro_r, pr_auc, f05, f1])
            case.extend([macro_p, macro_r, f1])
            # m_auc_f1, join_auc, auc, pr_mean, f05, f1

            # Patch for NNDR ONLY
            # case[7] = str(case[7])
            pr_macro_lst.append(case)

            # Printing the case and the time cosumed for calculation.
            print "append", case
            timel = tm.gmtime(tm.time() - start_tm)[3:6] + ((tm.time() - int(start_tm))*1000,)
            print "Time elapsed : %d:%d:%d:%d" % timel

            # print pr_macro_lst

            # Saving tables
            # np.savetxt(conf_mtrx_fname, conf_mtrx)

            # pr_tabel[np.where((pr_tabel==np.NaN))] = 0.0
            # np.savetxt(pr_tabel_fname, pr_table)

            # np.set_printoptions(precision=3, threshold=10000, suppress=True, linewidth=100)
            # print conf_mtrx
            # print pr_scores[:, :]"""

            # Closing HDF5 files
            h5d_fl1.close()

    # Getting the macro averaged P, R, F1, F05
    print pr_macro_lst
    pr_macroavgs = np.vstack(pr_macro_lst)

    """
    # # # Saving Resaults to the following file.
    prnt_case_od = coll.OrderedDict([
        ('critirion_idx', [-1, -2, -3]),  # -5, -6, -8
        ('dist', ['', 'MinMax', 'MIX']),  # '', 'MinMax',
        ('corpus', ['KI04']),  # , 'KI04', 'SANTINIS', 7Genres
        ('doc_rep', ['W3G'])
    ])
    """

    with open('/home/dimitrios/Openness_OCSVME_POS3G_KI04_2018_03_26_repeat_2nd.csv', 'w') as score_sf:

        json.dump(fp=score_sf, obj=pr_macroavgs[:, :].tolist())

        """
        for idx, dm, cr, dr in param_comb.ParamGridIter(prnt_case_od, 'list'):

            # Getting Combination.
            selctd_rows = pr_macroavgs[
                np.where(
                    (pr_macroavgs[:, 0] == dr) &
                    (pr_macroavgs[:, 1] == cr) &
                    (pr_macroavgs[:, 2] == dm)
                )
            ]

            # Printing... for f1 = -1 idx, f05 = -2 idx, auc = -4 idx

            if idx == -1:
                score_sf.write("F1 ")
            elif idx == -2:
                score_sf.write("F0.5 ")
            elif idx == -3:
                score_sf.write("PR MACRO AUC ")
            elif idx == -4:
                score_sf.write("ROC AUC ")
            elif idx == -5:
                score_sf.write("Mean ")
            elif idx == -6:
                score_sf.write("Marco JoinP AUCs / GeoMean ")
            elif idx == -7:
                score_sf.write("Marco AUCs JP F1 ")
            elif idx == -8:
                score_sf.write("Precision ")

            # Printing Optimal.
            values_float_array = np.array(selctd_rows[:, idx], dtype=np.float64)
            try:
                opt_idx = np.argmax(values_float_array)
                json.dump(fp=score_sf, obj=list(selctd_rows[opt_idx, :]))
                score_sf.write("\n")
            except:
                score_sf.write("Empty Array\n")
                print "Empty Array"
        """
