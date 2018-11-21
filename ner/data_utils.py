# encoding=utf-8
import codecs as _co
from rlog import _log_normal, _log_warning, _log_info, _log_error


_global_index = 0

SIGN_UNK = u'</s>'
SIGN_BLANK = u' '
SIGN_RETURN = u'\n'

NMT_UNK = '<unk>'
NMT_SOS = '<s>'
NMT_EOS = '</s>'
NMT_UNK_ID = 0
NMT_SOS_ID = 1
NMT_EOS_ID = 2


def compare_targets(file1, file2):
    if (file1 is None or file2 is None):
        _log_error('Invalid file name.')
        return

    lines1 = _co.open(file1, encoding='utf-8').readlines()
    lines2 = _co.open(file2, encoding='utf-8').readlines()

    length1 = len(lines1)
    length2 = len(lines2)

    if (length1 != length2):
        _log_error('Number of lines is not same between the two files.')
        return

    errors = 0.0
    for i in range(length1):
        if (lines1[i] != lines2[i]):
            errors += 1.0
    res = errors / length1

    _log_info('Precision: %.2f%%' % ((1.0 - res) * 100))
    return (1.0 - res)