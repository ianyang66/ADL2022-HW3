import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import json
import numpy as np

loss = []
steps = []
data = json.load(open('statistic/trainer_state.json'))

a=1
for i in range(0,141,3):
    loss.append(data['log_history'][i]['loss'])
    s=500*a
    a+=1
    steps.append(int(s))

plt.figure()
l = plt.plot(steps, np.array(loss), label="loss")

plt.legend(loc="upper right")
plt.xlabel('# of steps')
plt.ylabel('Loss')

plt.grid(ls ='--')
# set yaxis locator
y_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.savefig('result_loss.png')
plt.show()