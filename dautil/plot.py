import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_x_y_complex(x, y):
    '''x is a real array.
    y is a complex array.
    '''
    plt.figure()
    sns.tsplot(x).set_title('x')
    plt.figure()
    sns.tsplot(np.real(y)).set_title('real(y)')
    plt.figure()
    sns.tsplot(np.imag(y)).set_title('imag(y)')

    sns.jointplot(x=x, y=np.real(y), kind='reg')
    plt.title('real(y) vs. x')
    sns.jointplot(x=x, y=np.imag(y), kind='reg')
    plt.title('imag(y) vs. x')
    sns.jointplot(x=x, y=np.absolute(y), kind='reg')
    plt.title('abs(y) vs. x')
    sns.jointplot(x=x, y=np.angle(y), kind='reg')
    plt.title('arg(y) vs. x')
