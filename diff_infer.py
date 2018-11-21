import codecs as co


NA_FILE = '../ner_train_data/ner_train/test/Test.txt'
LF_FILE = './ner_corpus/test.lf.data_main'
INFER_FILE = './ner_corpus/test.infer.ner_main'

na_lines = co.open(NA_FILE, 'r', 'utf-8').readlines()
lf_lines = co.open(LF_FILE, 'r', 'utf-8').readlines()
infer_lines = co.open(INFER_FILE, 'r', 'utf-8').readlines()

assert len(na_lines) == len(lf_lines)
assert len(lf_lines) == len(na_lines)

data_len = len(na_lines)

final_res = ''
co.open('./diff_res.txt', 'w', 'utf-8').write('')
wf = co.open('./diff_res.txt', 'a', 'utf-8')

wf.write('ner diff result(na/lf/infer):\n\n---------------\n\n')

diff_count = 0

for i in range(data_len):
    lf = lf_lines[i].strip()
    infer = infer_lines[i].strip()
    if (lf != infer):
        na = na_lines[i].split(',')[0].strip()
        final_res = final_res + na + '\n' + lf + '\n' + infer + '\n-------\n'
        diff_count += 1
    if (diff_count > 0 and diff_count % 1000 == 0):
        wf.write(final_res)
        final_res = u''

if (final_res != u''):
    wf.write(final_res)

wf.flush()
wf.close()
