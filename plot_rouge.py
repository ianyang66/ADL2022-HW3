import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import json
import numpy as np

rouge1, rouge2, rougel = [0], [0], [0]
steps = [0]
data = json.load(open('statistic/trainer_state.json'))

a=1
for i in range(2,143,3):
    rouge1.append(data['log_history'][i]['rouge-1_f'])
    rouge2.append(data['log_history'][i]['rouge-2_f'])
    rougel.append(data['log_history'][i]['rouge-l_f'])
    s=500*a
    a+=1
    steps.append(int(s))

plt.figure()
r1 = plt.plot(steps, np.array(rouge1)*100, label="rouge-1")
r2 = plt.plot(steps, np.array(rouge2)*100, label="rouge-2")
rl = plt.plot(steps, np.array(rougel)*100, label="rouge-l")
plt.legend(loc="lower right")
plt.xlabel('# of steps')
plt.ylabel('F1 (%)')

plt.grid(ls ='--')
# set yaxis locator
y_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.savefig('result_f.png')
plt.show()