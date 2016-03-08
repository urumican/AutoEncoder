############
# Plot it
############
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

model = Sequential()
for encoder in encoders:
    model.add(encoder)
model.compile(loss='categorical_crossentropy', optimizer='Adam')

ae_test = model.predict(test_x)

colors = {0: 'b', 1: 'g', 2: 'r', 3:'c', 4:'m',
          5:'y', 6:'k', 7:'orange', 8:'darkgreen', 9:'maroon'}

markers = {0: 'o', 1: '+', 2: 'v', 3:'<', 4:'>',
          5:'^', 6:'s', 7:'p', 8:'*', 9:'x'}

plt.figure(figsize=(10, 10))
patches = []
for idx in xrange(0,300):
    point = ae_test[idx]
    label = test_y[idx]

    if label in [2,5,8,9]: #We skip these labels to make the plot clearer
        continue

    color = colors[label]
    marker = markers[label]
    line = plt.plot(point[0], point[1], color=color, marker=marker, markersize=8)

#plt.axis([-1.1, 1.1, -1.1, +1.1])