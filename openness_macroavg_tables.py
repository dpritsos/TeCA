# !/usr/bin/env python

import tables as tb
import numpy as np
import collections as coll
import time as tm
import json

import teca.analytics.paramcombs as param_comb
from prconf_tables import PRConf_table
from auc_tables import params_prauc_tables


# Run for all cases
if __name__ == '__main__':

    # Setting the file-name which will be used for reading the resaults-file and writing the...
    # ...calculations-file.
    rpath = '/home/dimitrios/RESULTS/NNDR_TF_RAW/'
    wpath = '/home/dimitrios/RESULTS/NNDR_TF_RAW/'
    fname = 'OpenSet_MU_NNDR_W3G_TF_15-2-3_V5-50-100k_SANTINIS_2018_10_07'

    # Parameters used for the experiments required for selecting specific or group of results
    kfolds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    case_od = coll.OrderedDict([
        ('terms_type', ['W3G']),
        ('vocab_size', [5000]),  # 5000, 10000, 50000
        # ('dims', [50]),  # 50, 100, 250, 500, 1000 100??
        # ('min_trm_fq', [3]),  # 3, 10
        # ('win_size', [8]),  # 3, 8, 20
        # ('algo', ['PV-DBOW']),
        # ('alpha', [0.025]),
        # ('min_alpha', [0.025]),
        # ('epochs', [10]),  # 1, 3, 10
        # ('decay', [0.002]),  # 0.002, 0.02
        # openness
        # ('uknw_ctgs_num', [1, 2, 3, 4, 5, 6, 7]),
        # ('uknw_ctgs_num_splt_itrs', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
        # RFSE
        # ('features_size', [1000]),
        # ('sim_func', ['minmax_sim']),
        # ('Sigma', [0.5, 0.7, 0.9]),
        # ('Iterations', [10, 50, 100]),
        # OCSVME
        # ('nu', [0.05, 0.07, 0.1, 0.15, 0.17, 0.3, 0.5, 0.7, 0.9]),
        # NNRD
        ('split_ptg', [0.5]),  # 0.7, 0.5
        ('ukwn_slt_ptg', [0.5]),  # 0.3, 0.5
        ('rt_lims_stp', [[0.4, 1.0, 0.2]]),
        ('lmda', [1.5]),
        # SVMRO
        # ('svm_type', ['oneclass']),
        # ('svm_type', ['binary']),
        # ('ll', [0.3, 0.8]),
        # ('c1_w', [0.3, 0.7]),
        # ('c2_w', [0.3, 0.7]),
        # ('mrgn_nw', [0.3, 0.7]),
        # ('mrgn_fw', [0.3, 0.7]),
        # SVMRO oneclass
        # ('nu', [0.05, 0.07, 0.1, 0.15, 0.17, 0.3, 0.5, 0.7, 0.9]),
        ('marked_uknw_ctg_lst', [12]),
        ('kfolds', ['']),
    ])
    """
    case_od = coll.OrderedDict([
        ('terms_type', ['W3G']),
        ('vocab_size', ['NA']), # 5000, 10000, 50000
        ('dims', [50, 100, 250, 500, 1000]),  # 50, 100, 250, 500, 1000 100??
        ('min_trm_fq', [3, 8, 10]),  # 3, 10
        ('win_size', [3, 8]),  # 3, 8, 20
        ('algo', ['PV-DBOW']),
        ('alpha', [0.025]),
        ('min_alpha', [0.025]),
        ('epochs', [1, 3, 10]),  # 1, 3, 10
        ('decay', [0.02, 0.002]),  # 0.002, 0.02
        # openness
        # ('uknw_ctgs_num', [1, 2, 3, 4, 5, 6, 7]),
        # ('uknw_ctgs_num_splt_itrs', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
        # RFSE
        # ('features_size', [1000]),
        # ('sim_func', ['minmax_sim']),
        # ('Sigma', [0.5, 0.7, 0.9]),
        # ('Iterations', [10, 50, 100]),
        # OCSVME
        # ('nu', [0.05, 0.07, 0.1, 0.15, 0.17, 0.3, 0.5, 0.7, 0.9]),
        # NNRD
        ('split_ptg', [0.5, 07]),  # 0.7, 0.5
        ('ukwn_slt_ptg', [0.3, 0.5]),  # 0.3, 0.5
        ('rt_lims_stp', [[0.8, 1.0, 0.2]]),
        ('lmda', [0.2, 0.5, 0.7]),
        # SVMRO
        # ('svm_type', ['oneclass']),
        # ('svm_type', ['binary']),
        # ('ll', [0.3, 0.8]),
        # ('c1_w', [0.3, 0.7]),
        # ('c2_w', [0.3, 0.7]),
        # ('mrgn_nw', [0.3, 0.7]),
        # ('mrgn_fw', [0.3, 0.7]),
        # SVMRO oneclass
        # ('nu', [0.05, 0.07, 0.1, 0.15, 0.17, 0.3, 0.5, 0.7, 0.9]),
        ('marked_uknw_ctg_lst', [12]),
        ('kfolds', ['']),
    ])
    """

    # FOR OPEN SET
    """
    case_od = coll.OrderedDict([
        ('terms_type', ['POS1G']),
        ('corpus', ['SANTINIS']),
        ('vocab_size', [43]),   # 5000, 10000, 50000, 100000, , 43, 1330, 16200,
        ('features_size', [4, 10, 20, 40]),  # , 5000, 10000 , 100, 500, 1000
        # 500, 1000, 5000, 10000, 50000, 90000
        # 4, 10, 20, 40, 100, 500, 1000, 5000, 10000
        ('sim_func', ['cosine_sim', 'minmax_sim']),  # , 'minmax_sim'
        ('Sigma', [0.5, 0.7, 0.9]),
        ('Iterations', [200, 300, 500, 1000]),
        # ('nu', [0.05, 0.07, 0.1, 0.15, 0.17, 0.3, 0.5, 0.7, 0.9]),
        ('marked_uknw_ctg_lst', [12]),
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

            if case[2] == 'MIX':
                # h5d_fl1 = tb.open_file(h5d_fl + '.h5', 'r')
                # h5d_fl2 = tb.open_file(h5d_fl + '_minmax.h5', 'r')
                mix = True

            elif case[2] == 'MinMax':
                # h5d_fl1 = tb.open_file(h5d_fl + '_minmax.h5', 'r')
                h5d_fl2 = None

            else:
                h5d_fl1 = tb.open_file(rpath + fname + '.h5', 'r')
                h5d_fl2 = None

            # ### Calculating the PR Curves ###
            # Preparing input for params_prauc_tables().
            param_od = coll.OrderedDict([
                ('terms_type', [case[0]]),
                ('vocab_size', [case[1]]),
                # ('dims', [case[2]]),
                # ('min_trm_fq', [case[3]]),
                # ('win_size', [case[4]]),
                # ('algo', [case[5]]),
                # ('alpha', [case[6]]),
                # ('min_alpha', [case[7]]),
                # ('epochs', [case[8]]),
                # ('decay', [case[9]]),
                # openness
                # ('uknw_ctgs_num', [case[11]]),
                # ('uknw_ctgs_num_splt_itrs', [case[12]]),
                # RFSE
                # ('features_size', [case[2]]),
                # ('sim_func', [case[12]]),
                # ('Sigma', [case[13]]),
                # ('Iterations', [case[14]]),
                # OCSVME
                # ('nu', [case[11]]),
                # NNRD
                ('split_ptg', [case[2]]),
                ('ukwn_slt_ptg', [case[3]]),
                ('rt_lims_stp', [case[4]]),
                ('lmda', [case[5]]),
                # SVMRO
                # ('svm_type', [case[11]]),
                # ('svm_type', [case[12]]),
                # ('ll', [case[13]]),
                # ('c1_w', [case[14]]),
                # ('c2_w', [case[15]]),
                # ('mrgn_nw', [case[16]]),
                # ('mrgn_fw', [case[17]]),
                # SVMRO oneclass
                # ('nu', [case[18]]),
                ('marked_uknw_ctg_lst', [case[6]]),
                ('kfolds', ['']),
            ])

            """
            param_od = coll.OrderedDict([
                ('terms_type', [case[0]]),
                ('vocab_size', [case[1]]),
                ('dims', [case[2]]),
                ('min_trm_fq', [case[3]]),
                ('win_size', [case[4]]),
                ('algo', [case[5]]),
                ('alpha', [case[6]]),
                ('min_alpha', [case[7]]),
                ('epochs', [case[8]]),
                ('decay', [case[9]]),
                # openness
                # ('uknw_ctgs_num', [case[11]]),
                # ('uknw_ctgs_num_splt_itrs', [case[12]]),
                # RFSE
                # ('features_size', [case[2]]),
                # ('sim_func', [case[12]]),
                # ('Sigma', [case[13]]),
                # ('Iterations', [case[14]]),
                # OCSVME
                # ('nu', [case[11]]),
                # NNRD
                ('split_ptg', [case[10]]),
                ('ukwn_slt_ptg', [case[11]]),
                ('rt_lims_stp', [case[12]]),
                ('lmda', [case[13]]),
                # SVMRO
                # ('svm_type', [case[11]]),
                # ('svm_type', [case[12]]),
                # ('ll', [case[13]]),
                # ('c1_w', [case[14]]),
                # ('c2_w', [case[15]]),
                # ('mrgn_nw', [case[16]]),
                # ('mrgn_fw', [case[17]]),
                # SVMRO oneclass
                # ('nu', [case[18]]),
                ('marked_uknw_ctg_lst', [case[14]]),
                ('kfolds', ['']),
            ])
            """
            pr_aucz_var_table = params_prauc_tables(
                h5d_fl1, h5d_fl2, 'multiclass_macro', kfolds, param_od, mix, strata=None, trec=False
            )

            # ### Calculating Marco Averaging of Precision, Recall, F1, F0.5 ###

            # Reformationg parametera path to be given properly to PRConf_table().
            print str(case_path.split('/')[1])
            # params_path = '/' + str(case_path.split('/')[1]) + '/' +\
            #     '/'.join(case_path.split('/')[3::])
            params_path = case_path
            # print pr_tabel_fname

            # Calculating the Precision and Recall Scores per Genre. Precision for a Genre is...
            # ...only calculated when there is at least one sample being counted for this Genre.
            # pre_cls_vect, rcl_cls_vect  = (1.0, 1.0)
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

            f05 = 1.25 * macro_p*macro_r / (0.25*macro_p+macro_r)
            if np.isnan(f05):
                f05 = 0

            # pr_mean = (macro_pr[0]+macro_pr[1]) / 2.0
            print pr_aucz_var_table

            pr_auc = pr_aucz_var_table[0, 8]  #15/16, 8
            # For RFSE is 8, for OCSVME 6, ECCE 5
            # Gensim For OCSVME 12, for RFSE 16, NNRD 15 or 16

            # maucs_num = len(m_aucz_var_table[0, 4::])  # For RFSE is 4, for OCSVME 3
            # m_auc_f1 = maucs_num / np.sum([1.0/mauc for mauc in m_aucz_var_table[0, 3::]])
            # if np.isnan(m_auc_f1):
            #     m_auc_f1 = 0

            # Macro Averaging AUCs per Genre.
            case.extend([macro_p, macro_r, pr_auc, f05, f1])
            # case.extend([macro_p, macro_r, f05, f1])
            # m_auc_f1, join_auc, auc, pr_mean, f05, f1

            # Patch for NNDR ONL
            print case
            case[4] = str(case[4])  # 4 or 11/12
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
            if case[2] == 'MIX':
                h5d_fl1.close()
                h5d_fl2.close()
            else:
                h5d_fl1.close()

    # Getting the macro averaged P, R, F1, F05
    print pr_macro_lst
    pr_macroavgs = np.vstack(pr_macro_lst)

    # # # Saving Resaults to the following file.
    with open(wpath + fname + '_NEW_AUC.csv', 'w') as score_sf:
        json.dump(fp=score_sf, obj=pr_macroavgs[:, :].tolist())
