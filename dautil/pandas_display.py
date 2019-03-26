from IPython.display import display
from ipywidgets import interact


def scrolling_dataframe(df, height=40, width=10):
    M = (df.shape[0] - 1) // height
    N = (df.shape[1] - 1) // width

    @interact(m=(0, M), n=(0, N))
    def f_display(m, n):
        return display(df.iloc[m * height:(m + 1) * height, n * width:(n + 1) * height])

    return f_display


def scrolling_dataframe_row(df, height=40):
    M = (df.shape[0] - 1) // height

    @interact(m=(0, M))
    def f_display(m):
        return display(df.iloc[m * height:(m + 1) * height])

    return f_display
