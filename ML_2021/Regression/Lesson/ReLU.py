import math
import matplotlib.pyplot as plt

# ReLU output = max(0,x)

def relu(x):
    if x < 0: return 0
    else: return x

def plot(px, py):
    plt.plot(px, py)
    # Get the current Axes instance on the current figure matching the given keyword args, or create one.
    # 例項化一個Axes物件
    ax = plt.gca()
    # 獲取上、左、底、右的座標脊例項
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    plt.show()

def main():
    x = []
    dx = -20
    while dx <= 20:
        x.append(dx)
        dx += 0.1

    px = [xv for xv in x]
    py = [relu(xv) for xv in x]
    plot(px, py)

if __name__ == "__main__":
    main()