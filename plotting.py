from matplotlib import pyplot as plt
from tabulate import tabulate

#x = [1e-06 ,0.00001, 0.00003, 0.00005,0.00008,  0.0001,0.0002, 0.0004,0.0006,0.0008, 0.001]#, 0.01, 0.1, 1.0]
x =  [-3, -2, -1, 0, 1, 2, 3]
means = []
stds = []
for i in x:
    with open('results/cutoff_20_loc_'+str(i)+'_scale_1e-06_laplace_noise.txt','r') as f:
        fl = f.readlines()[1].strip().split(' ')
        means.append(float(fl[0]))
        stds.append(float(fl[1][:-1]))
print(means)
xlist = list(range(len(x)))
plt.scatter(xlist,means)
plt.errorbar(xlist,means,stds,fmt='o')
plt.xticks(xlist,labels=x,rotation=45)
plt.xlabel('Mu location')
plt.ylabel('Error')
plt.savefig('results/laplace_noise_mu_changes.png')
plt.show()

print(tabulate({"Mu Location":x,"Mean Error":means,"Std Deviation":stds}, headers="keys",tablefmt="latex"))
