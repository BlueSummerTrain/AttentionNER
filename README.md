模型相关代码
data_preprocess.py　用于生成ner字典以及测试文件，并保存在nmt_corpus文件夹中。
create_vocab_data_for_nmt(input_file, output_file, keep_fre=0)函数用于生成字典，keep_fre控制删除词频小于keep_fre的字(注意关键字不要删了)
sperate_label_data用以分离数据和标签，并将标签保存。
以上两个函数在使用的时候，注意区分NER的任务，也就是注意正常实体提取以及宾语实体提取，通过‘main’与‘obj’区别

整个模型训练以及测试代码在ner目录下。
在训练前先执行data_preprocess.py生成字典与测试文件。
根据生成的字典数据修改超参hparams_modules.py
