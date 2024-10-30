import sys
import os
import glob

# 目前还有一些小缺陷，需要手动更改mif文件中的depth
def txt2mif(source_filename):
    # 生成目标文件名，将扩展名从 .txt 改为 .mif
    if source_filename.endswith('.txt'):
        destination_filename =source_filename[:-4] + '.mif'
    else:
        print('源文件必须以 .txt 结尾')
        return

    try:
        # 打开目标文件进行写入
        with open(destination_filename, 'w') as destination_file:
            # 写入固定内容
            destination_file.write("DEPTH=784;			--The size of data in bits\n")
            destination_file.write("WIDTH=8;			--The size of memory in words\n")
            destination_file.write("ADDRESS_RADIX=HEX;	--The radian for address values\n")
            destination_file.write("DATA_RADIX=HEX;		--The radian for data values\n")
            destination_file.write("CONTENT				--start of (address: data pairs)\n")
            destination_file.write("BEGIN\n")

            # 打开源文件进行读取
            with open(source_filename, 'r') as source_file:
                address = 0  # 初始地址
                # 逐行读取源文件的内容
                for line in source_file:
                    # 按空格分割成单个数字
                    numbers = line.split()
                    # 将每个数字写入目标文件，各占一行，前面加上地址
                    for number in numbers:
                        # 格式化地址为两位十六进制数
                        address_str = f"{address:03X}:"
                        # 写入地址和数字
                        destination_file.write(f"{address_str} {number};\n")
                        # 地址递增
                        address += 1

            # 写入文件结束标记
            destination_file.write("END;\n")

        print(f'内容已成功从 {source_filename} 复制到 {destination_filename}。')
    except FileNotFoundError:
        print(f'文件 {source_filename} 未找到。')
    except Exception as e:
        print(f'发生错误: {e}')


if __name__ == '__main__':
    # if len(sys.argv) != 2:
    #     print('用法: python script.py source_file.txt')
    # else:
    #     source_file = sys.argv[1]
    #     main(source_file)
    if not os.path.isdir('weight/txt'):
        print('不是一个有效的目录')
    
    txtfiles = glob.glob(os.path.join('weight/txt', '*.txt'))
    if not txtfiles:
        print(f'在目录weight/txt中未找到任何txt文件。')

    # 遍历所有txt文件并进行转换
    for txt_file in txtfiles:
        txt2mif(txt_file)
