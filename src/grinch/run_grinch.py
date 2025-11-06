from grinch import Grinch
import numpy as np

if __name__ == "__main__":
    point_labels = np.random.random_integers(0, 10, 100)
    vectors = np.random.random((100, 5)).astype(np.float32)
    grinch = Grinch(points=vectors)
    grinch.build_dendrogram()
    grinch.write_tree("tmp.tree.out", point_labels)
