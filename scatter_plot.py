import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 35}

plt.rc('font', **font)

PLANE = ["xy", "xz", "yz"][1]
df = pd.read_csv(f'{PLANE}_scores.csv', header=None)

names = df.iloc[:, 0].to_numpy()
c = df.iloc[:, 1].to_numpy()
x = df.iloc[:, 2].to_numpy()
y = df.iloc[:, 3].to_numpy()
score3 = df.iloc[:,4].to_numpy()

mask = score3 > 30

names = names[mask]
c = c[mask]
x = x[mask]
y = y[mask]
score3 = score3[mask]

print(f"# defects = {len(np.where(c == 2)[0])}")

norm = plt.Normalize(1,4)
cmap = plt.cm.RdYlGn

fig,ax = plt.subplots()
sc = plt.scatter(x,y,c=c, s=150, cmap=cmap, norm=norm)

e = sc.legend_elements(num=1)
e[0][0].set_markersize(20)
e[0][1].set_markersize(20)
nms = [[*e[0]], ["No Defect", "Defect (PLF)"]]
print(nms, "\n", sc.legend_elements(num=1))

legend1 = ax.legend(*nms, title="Classification")
ax.add_artist(legend1)

# HOVER
annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}, {}".format(" ".join([names[n] for n in ind["ind"]]), 
                           " ".join([str(round(score3[n], 1)) for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()
                
fig.canvas.mpl_connect("motion_notify_event", hover)
# HOVER

plt.xlabel("Score 1 (Convolution)")
plt.ylabel("Score 2 (Artifacts)")
#plt.ylim(0, 100)
#plt.xlim(0,20)
plt.show()