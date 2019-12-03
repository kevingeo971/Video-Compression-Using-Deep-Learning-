import time
from sklearn.cluster import AgglomerativeClustering

class ClusterModule:

    def __init__(self):
        self.ac = None

    def run(self, image_compressed, fps=10):
        n_samples = image_compressed.shape[0]
        self.ac = AgglomerativeClustering(n_clusters=n_samples // fps)
        start_time = time.time()
        self.ac.fit(image_compressed)
        print("Time to fit ", n_samples, ": ", time.time() - start_time)
        return self.ac.labels_

    def plot_distribution(self):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
        axs.hist(self.ac.labels_, bins=max(self.ac.labels_) + 1)
        plt.xlabel("Cluster Numbers")
        plt.ylabel("Number of datapoints")

