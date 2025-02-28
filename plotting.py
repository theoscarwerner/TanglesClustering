import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class BoxPlot:
    def __init__(self, df):
        self.df = df

    def time(self, log=False, x='Data-Size'):
        plt.figure(figsize=(15, 6))
        sns.boxplot(x=x, y='Time', hue='Model', data=self.df, palette='Set2')

        if log:
            plt.yscale('log')
            y_ticks = np.logspace(-3, 3, 7)

            for y in y_ticks:
                plt.axhline(y, color='gray', linestyle='--', alpha=0.5)

        plt.title('Time Distribution by Model and ' + x)
        plt.xlabel(x)
        plt.ylabel('Time')
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # plt.show()

    def nmi(self, x='Clusters', y='NMI'):
        plt.figure(figsize=(13, 6))
        sns.boxplot(x=x, y=y, hue='Model', data=self.df, palette='Set2')
        y_ticks = np.arange(0, 1.01, 0.05)

        for y_ in y_ticks:
            plt.axhline(y_, color='gray', linestyle='--', alpha=0.5)

        plt.title(f'{y} Distribution Compared to {x}')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        # plt.show()


def plot_cuts(data, cuts,
              title="Axis aligned features sorted by increasing order"):

    plot_size = int(np.ceil(np.sqrt(cuts.shape[1])))
    fig, axes = plt.subplots(nrows=plot_size, ncols=plot_size, figsize=(10, 10))
    # fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 3))
    for i, ax in enumerate(axes.flat):
        if i < cuts.shape[1]:
            ax.plot(data[cuts[:, i] > 0, 0], data[cuts[:, i] > 0, 1], "xr", markersize=2)
            ax.plot(data[cuts[:, i] <= 0, 0], data[cuts[:, i] <= 0, 1], "xb", markersize=2)
            ax.set_title(r"$P_{{{0}}}$".format(i+1), fontsize=16)
            ax.axis("off")
        else:
            ax.set_visible(False)
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    # plt.show()


def soft_predictions(data, preds):
    fig, ax = plt.subplots()
    scatter = ax.scatter(data[:, 0], data[:, 1],
                         c=preds @ np.arange(preds.shape[1]), cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, ax=ax)
    plt.show()


def plot_average_time(df):
    # Plots runtime.
    avg_time = df.groupby("Model")["Time"].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=avg_time, x="Model", y="Time", marker='o')
    plt.title("Average Time for Each Model")
    plt.xlabel("Model")
    plt.ylabel("Average Time (s)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_histogram_projection(node, mask, data, mean_left,
                              mean_right, direction_vector, hist, peaks):
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)

    c = ['#1E90FF' if node.left.p[mask][i] > (node.p[mask][i] / 2)
         else 'lightblue' for i in range(len(node.left.p[mask]))]

    plt.scatter(
        data[mask][:, 0], data[mask][:, 1], s=10, alpha=0.5, c=c)

    plt.scatter(mean_left[0], mean_left[1], color='red')
    plt.scatter(mean_right[0], mean_right[1], color='red')
    plt.plot([0, 1], [mean_left[1] + direction_vector[1] * (
                        0 - mean_left[0]) / direction_vector[0],
                      mean_left[1] + direction_vector[1] * (
                        1 - mean_left[0]) / direction_vector[0]], 'k--')

    # Draw arrows at right angles to the direction vector
    arrow_start = mean_left + direction_vector * 0.5
    arrow_end = arrow_start + np.array([-direction_vector[1], direction_vector[0]]) * 0.1
    plt.arrow(arrow_start[0], arrow_start[1],
              arrow_end[0] - arrow_start[0],
              arrow_end[1] - arrow_start[1],
              head_width=0.02, head_length=0.02, fc='k', ec='k')

    arrow_start = mean_right - direction_vector * 0.5
    arrow_end = arrow_start + np.array([direction_vector[1], -direction_vector[0]]) * 0.1

    plt.arrow(arrow_start[0], arrow_start[1],
              arrow_end[0] - arrow_start[0],
              arrow_end[1] - arrow_start[1],
              head_width=0.02, head_length=0.02, fc='k', ec='k')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.bar(np.arange(len(hist)), hist)
    plt.scatter(peaks, hist[peaks], color='red')
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])

    plt.savefig(f'projection_{abs(node.name)}.png', dpi=300, bbox_inches='tight')

    plt.show()
