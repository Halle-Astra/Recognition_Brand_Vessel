import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(-5,5,200)
nb = 1*(x>=0)
db = 1/(1+np.exp(-1*(x)))
db1 = 1/(1+np.exp(-3*(x)))
plt.plot(x,nb)
plt.plot(x,db)
plt.plot(x,db1)
plt.ylabel('$\mathcal{B}$')
plt.xlabel('$q-thresh_{\\rm binary}$')
plt.legend(['$\mathcal{B}_{\\rm orig}$','$\mathcal{B}_{\\rm DB},k=1$',
            '$\mathcal{B}_{\\rm DB},k=3$'])
plt.savefig('DB.png',dpi = 200)
plt.show()