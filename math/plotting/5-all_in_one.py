#!/usr/bin/env python3
"""Define function all_in_one."""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['axes.labelsize'] = 'x-small'
plt.rcParams['axes.titlesize'] = 'x-small'
plt.rcParams['legend.fontsize'] = 'x-small'


def all_in_one():
    """Plot all 5 previous graphs in one figure."""
    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    fig, axs = plt.subplot_mosaic([
        ['line', 'scatter'],
        ['log scale', 'two plots'],
        ['hist', 'hist']],
        layout="constrained",
        num='Tasks 0 to 4 as subplots'
        )
    fig.suptitle('All in One')
    ax00 = axs['line']
    ax00.plot(np.arange(0, 11), y0, linestyle='-', color='red')
    ax00.set_xlim(0, 10)

    ax01 = axs['scatter']
    ax01.scatter(x1, y1, color='magenta', marker='o')
    ax01.set_xlabel('Height (in)')
    ax01.set_ylabel('Weight (lbs)')
    ax01.set_title("Men's Height vs Weight")

    ax10 = axs['log scale']
    ax10.plot(x2, y2)
    ax10.set_yscale('log')
    ax10.set_xlabel('Time (years)')
    ax10.set_ylabel('Fraction Remaining')
    ax10.set_title('Exponential Decay of C-14')
    ax10.set_xlim(0, 28650)

    ax11 = axs['two plots']
    ax11.plot(x3, y31, linestyle='dashed', color='red', label='C-14')
    ax11.plot(x3, y32, '-g', label='Ra-226')
    ax11.legend(loc='upper right')
    ax11.set_xlabel('Time (years)')
    ax11.set_ylabel('Fraction Remaining')
    ax11.set_title('Exponential Decay of Radioactive Elements')
    ax11.set_xlim(0, 20_000)
    ax11.set_ylim(0, 1)

    ax2 = axs['hist']
    ax2.hist(student_grades,
             bins=np.arange(0, 101, 10, dtype=float).tolist(),
             edgecolor='black')
    ax2.set_ylim(0, 30)
    ax2.set_xlim(0, 100)
    ax2.set_xticks(np.arange(0, 101, 10))
    ax2.set_xlabel('Grades')
    ax2.set_ylabel('Number of Students')
    ax2.set_title('Project A')

    plt.show()
