import tensorflow as tf
import numpy as np
import csv
def weight_variable(shape):
    tf.compat.v1.set_random_seed(1)
    return tf.Variable(tf.random.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.0, shape=shape))

def reshape_weight(weight,k_w,k_h,in_ch,out_ch):

    re_weight = np.zeros([out_ch,in_ch,k_w,k_h])
    for i in range(k_w):
        for j in range(k_h):
            for k in range(in_ch):
                for q in range(out_ch):
                    re_weight[q][k][i][j] = weight[i][j][k][q]
    return re_weight

#初始化复原模型用变量
# conv1
W_conv1 = weight_variable([5, 5, 1, 3])
b_conv1 = bias_variable([3])

# conv2
W_conv2 = weight_variable([5, 5, 3, 3])
b_conv2 = bias_variable([3])

# full connection1
W_fc1 = weight_variable([4 * 4 * 3, 10])
b_fc1 = bias_variable([10])

#初始化存参用变量
# wconv1 = tf.zeros([5,5,1,32])
wconv1 = np.zeros([5,5,1,3])
bconv1 = np.zeros([3])
# wconv2 = tf.zeros([5,5,32,64])
wconv2 = np.zeros([5,5,3,3])
bconv2 = np.zeros([3])
wfc1 = np.zeros([4*4*3,10])
bfc1 = np.zeros([10])

# csv保存路径
cpkt_model_path = "model"
w_conv1_path = 'weight//w_conv1.csv'
re_w_conv1_path = 'weight//re_w_conv1.csv'
b_conv1_path = 'weight//b_conv1.csv'
w_conv2_path = 'weight//w_conv2.csv'
b_conv2_path = 'weight//b_conv2.csv'
w_fc1_path   = 'weight//w_fc1.csv'
b_fc1_path   = 'weight//b_fc1.csv'

q_w_conv1_path = 'weight//quan//w_conv1.csv'
q_b_conv1_path = 'weight//quan//qb_conv1.csv'
q_w_conv2_path = 'weight//quan//qw_conv2.csv'
q_b_conv2_path = 'weight//quan//qb_conv2.csv'
q_w_fc1_path   = 'weight//quan//qw_fc1.csv'
q_b_fc1_path   = 'weight//quan//qb_fc1.csv'

# 暂且使用归一化对每一层的参数进行单独缩放，缩放到-127-127
def quantize_weights(weights):
    # 假设weights是一个NumPy数组，scale_factor是缩放因子
    scale_factor = 127.0 / np.max(np.abs(weights))
    quantized_weights = np.round(weights * scale_factor).astype(np.int8)
    return quantized_weights


def to_8bit_signed_binary(num):
    """将整数转换为8位有符号二进制字符串"""
    if num < 0:
        # 如果是负数，则先取反再加1得到补码
        num = (1 << 8) + num
    return format(num & 0xFF, '08b')  # 确保是8位，并格式化为二进制字符串

def signed_binary(num, bits=8):
    """将整数转换为指定位数的有符号二进制字符串"""
    if num >= 0:
    # if num.any() >= 0:
        return format(num, '0' + str(bits) + 'b')
    else:
        # 对于负数，先取绝对值的二进制，然后取反加一得到补码
        return format((2 ** bits + num) % (2 ** bits), '0' + str(bits) + 'b')


if __name__ == "__main__":
    # 从cpkt中加载参数
    saver = tf.compat.v1.train.Saver(
        {'W_conv1': W_conv1, 'b_conv1': b_conv1, 'W_conv2': W_conv2, 'b_conv2': b_conv2,
        'W_fc1': W_fc1, 'b_fc1': b_fc1})
    with tf.compat.v1.Session() as sess:
        # 恢复模型
        module_file = tf.train.latest_checkpoint(cpkt_model_path)
        saver.restore(sess, module_file)

        # 提取参数值并进行reshape
        wconv1 = sess.run(W_conv1)
        re_wconv1 = reshape_weight(wconv1,5,5,1,3)
        bconv1 = sess.run(b_conv1)
        wconv2 = sess.run(W_conv2)
        re_wconv2 = reshape_weight(wconv2,5,5,3,3)
        bconv2 = sess.run(b_conv2)
        wfc1 = sess.run(W_fc1)
        bfc1 = sess.run(b_fc1)

        # 量化
        quan_wconv1 = quantize_weights(re_wconv1)
        quan_bconv1 = quantize_weights(bconv1)
        quan_wconv2 = quantize_weights(re_wconv2)
        quan_bconv2 = quantize_weights(bconv2)
        quan_wfc1   = quantize_weights(wfc1)
        quan_bfc1   = quantize_weights(bfc1)

        # # 保存到CSV文件
        # with open(w_conv1_path, 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     # 写入W_conv1权重
        #     writer.writerow("conv1 weight")
        #     for row in wconv1:
        #         writer.writerow(row)
        #     # 写入一个空行作为分隔
        #     writer.writerow([])
        #
        # # with open(re_w_conv1_path, 'w', newline='') as csvfile:
        # #     writer = csv.writer(csvfile)
        # #     # 写入W_conv1权重
        # #     writer.writerow("re_conv1 weight")
        # #     for row in re_wconv1:
        # #         writer.writerow(row)
        # #     # 写入一个空行作为分隔
        # #     writer.writerow([])
        #
        # with open(b_conv1_path, 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow("conv1 bias")
        #     # 写入b_conv1偏置
        #     writer.writerow(bconv1)
        # with open(w_conv2_path, 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     # 写入W_conv1权重
        #     writer.writerow("conv2 weight")
        #     for row in wconv2:
        #         writer.writerow(row)
        #     # 写入一个空行作为分隔
        #     writer.writerow([])
        # with open(b_conv2_path, 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow("conv1 bias")
        #     # 写入b_conv1偏置
        #     writer.writerow(bconv2)
        #
        #     # 写入W_conv1权重
        # with open(w_fc1_path, 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow("fc1 weight")
        #     for row in wfc1:
        #         writer.writerow(row)
        #     # 写入一个空行作为分隔
        #     writer.writerow([])
        # with open(b_fc1_path, 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow("fc1 bias")
        #     # 写入b_conv1偏置
        #     writer.writerow(bfc1)
        #
        #     # 写入W_conv1权重
        # with open(w_fc2_path, 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow("fc2 weight")
        #     for row in wfc2:
        #         writer.writerow(row)
        #     # 写入一个空行作为分隔
        #     writer.writerow([])
        # with open(b_fc2_path, 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow("fc2 bias")
        #     # 写入b_conv1偏置
        #     writer.writerow(bfc2)


        # 量化后保存到csv
        with open(q_w_conv1_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # 写入W_conv1权重
            writer.writerow("conv1 weight")
            for row in quan_wconv1:
                writer.writerow(row)
            # 写入一个空行作为分隔
            writer.writerow([])
        with open(q_b_conv1_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow("conv1 bias")
            # 写入b_conv1偏置
            writer.writerow(quan_bconv1)
        with open(q_w_conv2_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # 写入W_conv1权重
            writer.writerow("conv2 weight")
            for row in quan_wconv2:
                writer.writerow(row)
            # 写入一个空行作为分隔
            writer.writerow([])
        with open(q_b_conv2_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow("conv1 bias")
            # 写入b_conv1偏置
            writer.writerow(quan_bconv2)

            # 写入W_conv1权重
        with open(q_w_fc1_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow("fc1 weight")
            for row in quan_wfc1:
                writer.writerow(row)
            # 写入一个空行作为分隔
            writer.writerow([])
        with open(q_b_fc1_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow("fc1 bias")
            # 写入b_conv1偏置
            writer.writerow(quan_bfc1)


        # 量化后保存到txt
        for i in range(quan_wconv1.shape[0]):
            for j in range(quan_wconv1.shape[1]):
                matrix = quan_wconv1[i][j]
                # matrix_str = '\n'.join([' '.join(map(str, row)) for row in matrix])
                # matrix_str = '\n'.join(['\n'.join(format(x, '08b') for x in row) for row in matrix])
                matrix_bin_str = '\n'.join([
                '\n'.join(signed_binary(x) for x in row) for row in matrix
                ])
                # matrix_str = ' '.join(str(x) for x in matrix)
                # 保存到txt文件，文件名包含i和j的索引
                with open(f'weight//txt//conv1_w_{i}_{j}.txt', 'w') as f:
                    f.write(matrix_bin_str)


        matrix = quan_bconv1
        # 将5x5矩阵转换为字符串，每行元素之间用空格分隔，每行结束后添加换行符
        # matrix_str = '\n'.join([' '.join(map(str, row)) for row in matrix])
        # matrix_str = '\n'.join(str(x) for x in matrix)
        matrix_str = '\n'.join(signed_binary(x) for x in matrix)
        # 保存到txt文件，文件名包含i和j的索引
        with open(f'weight//txt//conv1_b_{i}.txt', 'w') as f:
            f.write(matrix_str)


        for i in range(quan_wconv2.shape[0]):
            for j in range(quan_wconv2.shape[1]):
                matrix = quan_wconv2[i][j]
                # 将5x5矩阵转换为字符串，每行元素之间用空格分隔，每行结束后添加换行符
                matrix_bin_str = '\n'.join([
                    '\n'.join(signed_binary(x) for x in row) for row in matrix
                ])
                # matrix_str = '\n'.join([' '.join(map(str, row)) for row in matrix])
                # matrix_str = ' '.join(str(x) for x in matrix)
                # 保存到txt文件，文件名包含i和j的索引
                with open(f'weight//txt//conv2_{i}_{j}.txt', 'w') as f:
                    f.write(matrix_str)

        matrix = quan_bconv2
        # 将5x5矩阵转换为字符串，每行元素之间用空格分隔，每行结束后添加换行符
        # matrix_str = '\n'.join([' '.join(map(str, row)) for row in matrix])
        matrix_str = ' '.join(signed_binary(x) for x in matrix)
        # 保存到txt文件，文件名包含i和j的索引
        with open(f'weight//txt//conv2_b_{i}.txt', 'w') as f:
            f.write(matrix_str)

        np.savetxt('weight//txt//wfc1.txt', quan_wfc1, fmt='%d', delimiter=' ')
        # np.savetxt('weight//txt//bfc1.txt', quan_bfc1, fmt='%d', delimiter=' ')



    # 打开文件以写入二进制数据
    with open('weight/txt/fc1_w.txt', 'w') as f:
        for row in quan_wfc1:
            # 将每一行中的每个数转换为8位有符号二进制字符串，并写入文件，每个数后面跟一个换行符
            for num in row:
                f.write(to_8bit_signed_binary(num) + '\n')
            # 可选：在每行数字之后添加一个空行以增加可读性
            # f.write('\n')


        matrix = quan_bfc1
        # 将5x5矩阵转换为字符串，每行元素之间用空格分隔，每行结束后添加换行符
        # matrix_str = '\n'.join([' '.join(map(str, row)) for row in matrix])
        # matrix_str = '\n'.join(str(x) for x in matrix)
        matrix_str = '\n'.join(signed_binary(x) for x in matrix)
        # 保存到txt文件，文件名包含i和j的索引
        with open(f'weight//txt//fc1_b.txt', 'w') as f:
            f.write(matrix_str)


