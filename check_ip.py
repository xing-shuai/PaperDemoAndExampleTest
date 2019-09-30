import os

ava_ips = []
allocated_ips = []
for i in range(2, 255):
    command = os.system('ping -c 1 10.21.243.' + str(i))
    if command == 0:
        allocated_ips.append(i)
        pass  # Sucess
    else:
        ava_ips.append(i)
        # print(str(i))

print('allocated_ips', allocated_ips)
print('ava_ips', ava_ips)
