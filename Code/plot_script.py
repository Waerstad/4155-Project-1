import matplotlib
from matplotlib.ticker import MultipleLocator

MM_TO_INCH = 1/25.4

matplotlib.rc("text", usetex=False)
plt.rc('lines', linewidth=1.0)
plt.rc("figure", autolayout=True)
plt.rc("legend", fontsize=6)
plt.rc("font", family="serif", size=8)
plt.figure(figsize=(86.78*MM_TO_INCH, 70*MM_TO_INCH))
plt.ticklabel_format(axis = "y", style="sci")
plt.gca().xaxis.set_major_locator(MultipleLocator(base=2, offset=-1))   # Set x axis

plt.plot(x, y, ".-", label="label")
plt.legend()
plt.xlabel("y")
plt.ylabel("x")
plt.grid(True, which="both", linestyle="--", linewidth=0.25)
#plt.savefig("name.pdf", bbox_inches="tight")
plt.show()
plt.close()