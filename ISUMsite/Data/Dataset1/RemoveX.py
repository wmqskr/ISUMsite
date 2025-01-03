def remove_x_from_file(filename):
    # 读取原始文件并删除所有 'X' 字符
    with open(filename, 'r') as file:
        content = file.read()
        content_without_x = content.replace('X', '')
    # 将处理后的数据写入临时文件
    with open('17807negative_samples_cleaned.txt', 'w') as cleaned_file:
        cleaned_file.write(content_without_x)

remove_x_from_file('17807negative_samples.txt')