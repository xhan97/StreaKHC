"""
Copyright (C) 2017 University of Massachusetts Amherst.
This file is part of "xcluster"
http://github.com/iesl/xcluster
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import math

import numpy as np
from numba import jit


@jit(nopython=True)
def _fast_dot(x, y):
    """Compute the dot product of x and y using numba.

      Args:
      x - a numpy vector (or list).
      y - a numpy vector (or list).

      Returns:
      x_T.y
      """
    return np.dot(x, y)


def _fast_normalize_dot(x, y):
    """Compute the dot product of x and y using numba.

      Args:
      x - a numpy vector (or list).
      y - a numpy vector (or list).
      t - an integel

      Returns:
      Normalized x_T.y
      """

    return _fast_dot(x, y) / (math.sqrt(_fast_dot(x, x)) * (math.sqrt(_fast_dot(y, y))))


try:
    from Queue import Queue
except:
    pass

try:
    from queue import Queue
except:
    pass


def serliaze_tree_to_file_with_node_ds(root, fn):
    with open(fn, "w") as fout:
        queue = Queue()
        queue.put(root)
        while not queue.empty():
            curr_node = queue.get()
            curr_node_id = curr_node.pts[0][1] if curr_node.is_leaf() else curr_node.id
            sibling_node_id = "None"
            if curr_node.parent:
                sibling_node_id = (
                    curr_node.siblings()[0].pts[0][1]
                    if curr_node.siblings()[0].is_leaf()
                    else curr_node.siblings()[0].id
                )
            ns = node_similarity(curr_node) if curr_node.parent else "None"
            fout.write(
                "%s\t%s\t%s\t%s\t%%.2f\t%s\n"
                % (
                    curr_node_id,
                    sibling_node_id,
                    curr_node.parent.id if curr_node.parent else "None",
                    ns,
                    curr_node.depth() if curr_node.parent else 0,
                    curr_node.pts[0][0] if curr_node.is_leaf() else "None",
                )
            )
            for c in curr_node.children:
                queue.put(c)


def node_similarity(curr_node):
    """Compute the similarity between a point x and its sibing.
        Args:
        x - a numpy array of floats.
        Returns:
        A float representing the lower bound.
        """
    sibling_node = curr_node.siblings()[0]
    # if curr_node.is_leaf() and sibling_node.is_leaf():
    #     return _fast_dot(curr_node.ikv, sibling_node.ikv)
    # elif curr_node.is_leaf():
    #     return _fast_normalize_dot(curr_node.ikv, sibling_node.ikv)
    # elif sibling_node.is_leaf():
    #     return _fast_normalize_dot(sibling_node.ikv, curr_node.ikv)
    # else:
    return _fast_normalize_dot(curr_node.ikv, sibling_node.ikv)


# def getDistance(curr_node):
#     sibling = curr_node.siblings()[0]
#     curr_point_count = curr_node.point_counter
#     sibling_point_count = sibling.point_counter

#     distance = 2*(200 - (_fast_dot(sibling.ikv, curr_node.ikv) /
#                   (curr_point_count*sibling_point_count)))

#     return distance


def serliaze_tree_to_file(root, fn):
    with open(fn, "w") as fout:
        queue = Queue()
        queue.put(root)
        while not queue.empty():
            curr_node = queue.get()
            fout.write(
                "%s\t%s\t%s\t%s\n"
                % (
                    curr_node.id,
                    curr_node.parent.id if curr_node.parent else "None",
                    curr_node.pts[0][1] if curr_node.is_leaf() else "None",
                    len(curr_node.pts),
                )
            )
            for c in curr_node.children:
                queue.put(c)


def serliaze_collapsed_tree_to_file_with_point_ids(root, fn):
    with open(fn, "w") as fout:
        queue = Queue()
        queue.put(root)
        while not queue.empty():
            curr_node = queue.get()
            curr_node_id = (
                curr_node.pts[0][2]
                if curr_node.is_leaf() and not curr_node.is_collapsed
                else curr_node.id
            )
            fout.write(
                "%s\t%s\t%s\n"
                % (
                    curr_node_id,
                    curr_node.parent.id if curr_node.parent else "None",
                    curr_node.pts[0][1]
                    if curr_node.is_leaf() and not curr_node.is_collapsed
                    else "None",
                )
            )
            for c in curr_node.children:
                queue.put(c)
            if curr_node.collapsed_leaves is not None:
                for c in curr_node.collapsed_leaves:
                    queue.put(c)
