import matplotlib.pyplot as plt
import csv

class Logger():
    def __init__(self):
        self.data = {}

    def store(self, entry):
        for key in entry:
            if key in self.data:
                self.data[key].append(entry[key])
            else:
                self.data[key] = [entry[key]]

    def save_plot(self, out_path, key, y_label=None):
        '''
        Saves the plot of the data in the dictionary with a specific key.
        If key is a list, saves a single plot with both measures.
        '''
        if isinstance(key, str):
            keys = [key]
        elif isinstance(key, list):
            keys = key

        if y_label is None: y_label = keys[0]

        xpoints = range(1, len(self.data[keys[0]]) + 1)
        for key in keys:
            ypoints = self.data[key]
            plt.plot(xpoints, ypoints, label=key)

        plt.xlabel("Epoch")
        plt.ylabel(y_label)

        if len(keys) > 1: plt.legend(loc="upper right")

        plt.savefig(out_path)
        plt.close()


    def save_plots(self, out_dir, keys="all"):
        if keys == "all": keys = self.data.keys()

        for key in keys:
            self.save_plot(f"{out_dir}/{key}.png", key)

    def save_csv(self, out_path):
        with open(out_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.data.keys()) # header

            for row in zip(*self.data.values()):
                writer.writerow(row)