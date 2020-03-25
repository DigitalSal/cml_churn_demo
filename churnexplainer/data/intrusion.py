import os
import pandas as pd

from churnexplainer.utils import data_dir
from churnexplainer.data import utils

intrusioncsvpath = os.path.join(data_dir, 'raw', 'KDDTrain+.txt')

idcol = 'connection_id'
labelcol = 'intrusion'
cols = (
    ('duration', False),
    ('protocol_type', True),
    ('service', True),
    ('flag', True),
    ('src_bytes', False),
    ('dst_bytes', False),
    ('land', False),
    ('wrong_fragment', False),
    ('urgent', False),
    ('hot', False),
    ('num_failed_logins', False),
    ('logged_in', False),
    ('num_compromised', False),
    ('root_shell', False),
    ('su_attempted', False),
    ('num_root', False),
    ('num_file_creations', False),
    ('num_shells', False),
    ('num_access_files', False),
    ('num_outbound_cmds', False),
    ('is_host_login', False),
    ('is_guest_login', False),
    ('count', False),
    ('srv_count', False),
    ('serror_rate', False),
    ('srv_serror_rate', False),
    ('rerror_rate', False),
    ('srv_rerror_rate', False),
    ('same_srv_rate', False),
    ('diff_srv_rate', False),
    ('srv_diff_host_rate', False),
    ('dst_host_count', False),
    ('dst_host_srv_count', False),
    ('dst_host_same_srv_rate', False),
    ('dst_host_diff_srv_rate', False),
    ('dst_host_same_src_port_rate', False),
    ('dst_host_srv_diff_host_rate', False),
    ('dst_host_serror_rate', False),
    ('dst_host_srv_serror_rate', False),
    ('dst_host_rerror_rate', False),
    ('dst_host_srv_rerror_rate', False),
    (labelcol, True),
    # metadata about how difficult this sample is to classify
    ('difficulty', True)
)


def load_dataset():
    df = pd.read_csv(intrusioncsvpath, names=[c for c, iscat in cols])
    df.index.name = idcol
    df = utils.categorize(df, cols)
    df[labelcol] = (df[labelcol] != 'normal')  # abnormal connection
    df = df.drop(cols[-1][0], axis=1)          # last column is not feature
    return utils.splitdf(df, labelcol)
