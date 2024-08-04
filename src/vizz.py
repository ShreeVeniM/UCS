import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_plot(fig, filename):
    os.makedirs('charts', exist_ok=True)
    fig.savefig(os.path.join('charts', filename), dpi=300)
    plt.close(fig)

def pairplot(df):
    try:
        fig = sns.pairplot(df[['Age','Annual_Income','Spending_Score']])
        save_plot(fig.fig, 'pairplot.png')
    except Exception as e:
        print(f"Error creating pairplot: {e}")

def scatter_plot(df, x, y, hue):
    try:
        fig, ax = plt.subplots()
        sns.scatterplot(x=x, y=y, data=df, hue=hue, palette='colorblind', ax=ax)
        save_plot(fig, 'scatterplot.png')
    except Exception as e:
        print(f"Error creating scatter plot: {e}")

def elbow_plot(k, WCSS):
    try:
        fig, ax = plt.subplots()
        ax.plot(k, WCSS)
        ax.set_xlabel('No. of clusters')
        ax.set_ylabel('WSS Score')
        ax.set_title('Elbow Plot')
        save_plot(fig, 'elbow_plot.png')
    except Exception as e:
        print(f"Error creating elbow plot: {e}")

def silhouette_plot(k, ss):
    try:
        fig, ax = plt.subplots()
        ax.plot(k, ss)
        ax.set_xlabel('No. of clusters')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Silhouette Plot')
        save_plot(fig, 'silhouette_plot.png')
    except Exception as e:
        print(f"Error creating silhouette plot: {e}")
