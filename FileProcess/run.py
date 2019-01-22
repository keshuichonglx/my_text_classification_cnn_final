import codecs


def concat_new_old_to_train():
    file1 = "E:\\competition\\SMP2018\\my_text_classification_cnn_old_for_com\\res\\data\\training_new.txt"
    file2 = "E:\\competition\\SMP2018\\my_text_classification_cnn_old_for_com\\res\\data\\training_old.txt"

    with open(file1, 'r', encoding='utf-8', errors='ignore') as f:
        ltRet1 = [line for line in f]

    with open(file2, 'r', encoding='utf-8', errors='ignore') as f:
        ltRet2 = [line for line in f]

    save_path = "E:\\competition\\SMP2018\\my_text_classification_cnn_old_for_com\\res\\data\\training.txt"
    with codecs.open(save_path, "w", "utf-8")as f:
        f.write("".join(ltRet1+ltRet2))
    print('完成: ' + save_path)

if __name__ == "__main__":
    concat_new_old_to_train()