import os
import pandas as pd

from collections import defaultdict

from churnexplainer.utils import data_dir
from churnexplainer.data import utils

idcol = 'id'
labelcol = 'not_repaid'
cols = (
     # ('id', False),
     # ('member_id', False),
     ('loan_amnt', False),
     # ('funded_amnt', False),
     # ('funded_amnt_inv', False),
     ('term', True),
     ('int_rate', False),
     # ('installment', ),
     # ('grade', True),
     # ('sub_grade', True),
     # ('emp_title', False),
     # ('emp_length', True),
     # ('home_ownership', True),
     ('annual_inc', False),
     # ('verification_status', False),
     # ('issue_d', False)                   datetime,
     # ('loan_status', True)                ~= target label,
     # ('pymnt_plan', ),
     # ('url', False)                       string,
     # ('desc', False)                      string,
     ('purpose', True),
     # ('title', True)                      string version of purpose,
     # 'zip_code',,
     # ('addr_state', True),
     ('dti', False),
     # ('delinq_2yrs', True),
     # ('earliest_cr_line', False)          datetime,
     # ('fico_range_low', False)            degenerate with grade,
     # ('fico_range_high', False)           degenerate with grade,
     # ('inq_last_6mths', ),
     # ('mths_since_last_delinq', ),
     # ('mths_since_last_record', ),
     # ('open_acc', ),
     # ('pub_rec', ),
     # ('revol_bal', ),
     ('revol_util', False),
     # ('total_acc', ),
     # ('initial_list_status', ),
     # ('out_prncp', ),
     # ('out_prncp_inv', ),
     # ('total_pymnt', ),
     # ('total_pymnt_inv', ),
     # ('total_rec_prncp', ),
     # ('total_rec_int', ),
     # ('total_rec_late_fee', ),
     # ('recoveries', ),
     # ('collection_recovery_fee', ),
     # ('last_pymnt_d', ),
     # ('last_pymnt_amnt', ),
     # ('next_pymnt_d', ),
     # ('last_credit_pull_d', ),
     # ('last_fico_range_high', ),
     # ('last_fico_range_low', ),
     # ('collections_12_mths_ex_med', ),
     # ('mths_since_last_major_derog', ),
     # ('policy_code', ),
     # ('application_type', ),
     # ('annual_inc_joint', ),
     # ('dti_joint', ),
     # ('verification_status_joint', ),
     # ('acc_now_delinq', ),
     # ('tot_coll_amt', ),
     # ('tot_cur_bal', ),
     # ('open_acc_6m', ),
     # ('open_il_6m', ),
     # ('open_il_12m', ),
     # ('open_il_24m', ),
     # ('mths_since_rcnt_il', ),
     # ('total_bal_il', ),
     # ('il_util', ),
     # ('open_rv_12m', ),
     # ('open_rv_24m', ),
     # ('max_bal_bc', ),
     # ('all_util', ),
     # ('total_rev_hi_lim', ),
     # ('inq_fi', ),
     # ('total_cu_tl', ),
     # ('inq_last_12m', ),
     # ('acc_open_past_24mths', ),
     # ('avg_cur_bal', ),
     # ('bc_open_to_buy', ),
     # ('bc_util', ),
     # ('chargeoff_within_12_mths', ),
     # ('delinq_amnt', ),
     # ('mo_sin_old_il_acct', ),
     # ('mo_sin_old_rev_tl_op', ),
     # ('mo_sin_rcnt_rev_tl_op', ),
     # ('mo_sin_rcnt_tl', ),
     # ('mort_acc', ),
     # ('mths_since_recent_bc', ),
     # ('mths_since_recent_bc_dlq', ),
     # ('mths_since_recent_inq', ),
     # ('mths_since_recent_revol_delinq', ),
     # ('num_accts_ever_120_pd', ),
     # ('num_actv_bc_tl', ),
     # ('num_actv_rev_tl', ),
     # ('num_bc_sats', ),
     # ('num_bc_tl', ),
     # ('num_il_tl', ),
     # ('num_op_rev_tl', ),
     # ('num_rev_accts', ),
     # ('num_rev_tl_bal_gt_0', ),
     # ('num_sats', ),
     # ('num_tl_120dpd_2m', ),
     # ('num_tl_30dpd', ),
     # ('num_tl_90g_dpd_24m', ),
     # ('num_tl_op_past_12m', ),
     # ('pct_tl_nvr_dlq', ),
     # ('percent_bc_gt_75', ),
     # ('pub_rec_bankruptcies', ),
     # ('tax_liens', ),
     # ('tot_hi_cred_lim', ),
     # ('total_bal_ex_mort', ),
     # ('total_bc_limit', ),
     # ('total_il_high_credit_limit', ),
     # ('frac_repaid', ),
     ('not_repaid', False),
 )


csv_files = ['LoanStats3a_securev1.csv.zip',
             'LoanStats3b_securev1.csv.zip',
             'LoanStats3c_securev1.csv.zip',
             'LoanStats3d_securev1.csv.zip',
             'LoanStats_securev1_2016Q1.csv.zip']

read_csv_defaults = {'skiprows': 1, 'skipfooter': 3, 'engine': 'python'}
read_csv_args = defaultdict(lambda: read_csv_defaults)
# Merge dict, python 3.5+
read_csv_args['LoanStats3a_securev1.csv.zip'] = {**read_csv_defaults,
                                                 **{'skipfooter': 2755}}


def read_raw_data():
    print("Loading CSVs")
    loans = pd.DataFrame()
    for f in csv_files:
        fpath = os.path.join(data_dir, 'raw', 'lendingclub', f)
        these_loans = pd.read_csv(fpath, **read_csv_args[f])
        loans = loans.append(these_loans, ignore_index=True)
        print(f, len(these_loans))
    print("Read {} loans".format(len(loans)))
    return loans


def parse_term(loans):
    print("Parsing terms")
    loans['term'] = loans['term'].str.split().str.get(0).astype(int)
    return loans


def parse_percent(loans):
    print("Parsing percentages")
    percentage_columns = ['int_rate', 'revol_util']
    for c in percentage_columns:
        loans[c] = loans[c].str.split('%').str.get(0).astype(float)
    return loans


def remove_unfully_paid(loans):
    bad = loans.query('loan_status == "Fully Paid" and frac_repaid < 1').index
    print(
        "Removing {} 'Fully Paid' loans with frac_repaid < 1".format(len(bad))
    )
    return loans.drop(bad)


def add_frac_repaid(loans):
    print("Adding frac_repaid column")
    loans['frac_repaid'] = loans['total_rec_prncp']/loans['loan_amnt']
    return loans


def remove_overpaid(loans):
    bad = loans.query('frac_repaid > 1').index
    print("Removing {} loans with frac_repaid > 1".format(len(bad)))
    return loans.drop(bad)


def remove_incomplete(loans):
    completed = loans['loan_status'].isin(['Fully Paid', 'Charged Off'])
    print("Removing {} incomplete loans".format((~completed).sum()))
    return loans[completed]


def remove_missing_revol_util(loans):
    missing = loans['revol_util'].isnull()
    print("Removing {} loans with missing revol_util".format(missing.sum()))
    return loans[~missing]


def add_not_repaid(loans):
    print("Adding repaid column")
    loans['not_repaid'] = (loans['frac_repaid'] != 1)
    return loans


def load_dataset():
    try:
        loans = utils.load_processed_dataset('loans')
    except IOError:
        print('Not found. Regenerating...')
        loans = read_raw_data()
        loans = loans.set_index(idcol)
        loans = remove_incomplete(loans)
        loans = remove_missing_revol_util(loans)

        loans = add_frac_repaid(loans)
        loans = remove_unfully_paid(loans)
        loans = remove_overpaid(loans)
        loans = add_not_repaid(loans)

        loans = parse_term(loans)
        loans = parse_percent(loans)

        loans = utils.categorize(loans, cols)
        loans = utils.drop_non_features(loans, cols)
        utils.save_processed_dataset(loans, 'loans')

    return utils.splitdf(loans, labelcol)
