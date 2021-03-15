from matplotlib import pyplot as plt

x = [0.0001,0.0002,0.0003,0.0004,0.0005]
means = []
stds = []
for i in x:
    with open('cutoff_20_normal_'+str(i)+'.txt','r') as f:
        fl = f.readlines()[1].strip().split(' ')
        means.append(float(fl[0]))
        stds.append(float(fl[1][:-1]))
print(means)
plt.errorbar(x,means,stds)
plt.xlabel('sigma')
plt.ylabel('mean error')
plt.show()
