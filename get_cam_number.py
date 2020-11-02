import os
import glob
import re

li = os.listdir('/dev/')

#print([i[:5] == 'video' for i in li])

# rst = []
# for i in li:
#     if i[:5] == 'video':
#         rst.append(i)

# print(rst)


def getDevicesList():
    devices_list = []

    result = os.popen('v4l2-ctl --list-devices').read()
    result_lists = result.split("\n\n")
    for result_list in result_lists:
        if result_list != '':
            result_list_2 = result_list.split('\n\t')
            devices_list.append(result_list_2[1])
    return devices_list

b = list(map(lambda x : x[-1], getDevicesList()))

print(f'현재 활성화 되어있는 카메라는 {b} 입니다.')

