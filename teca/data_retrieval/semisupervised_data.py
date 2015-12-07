"""
    This module includes RFSE data retrieval functions.
    Details can be found in each funciton separatly.

    Aurthor: Dimitrios Pritos

"""

import sys
sys.path.append('../../teca')

import numpy as np


def get_predictions(res_h5file, params_path, class_tag=None):
    """Retrieval functions for the date returned from the Semi-Supervised EM/Kmeans Clustering.

    Returns the cluster tags after the clustering procedure together with the expected classes of
    the documents and the clustering parameters. Additionally a specific class-tag can be given as
    argument for retrieving the respective clustering outcome for this documents.

   Arguments
   ---------
        res_h5file: The HD5 file object where the results are expected to be located.

        params_path: The path in HD5 where the Cluster and Class tags are located.
            The HD5 file is expected to have a path where in each node there is a parameter value
            used for the experiments.

        genre_tag: The number (integer) which is the tag of the class tag to be retrieved.
            Default value is 'None'.

    Output
    ------
        clstr_y: The outcome of the semi-supervised clustering procedure.
        clss_y: The original/known class of the documents has been clustered. Probably for Bcubed
            precision recall calculations.
        clstr_params: The parameters of the semi-supervised clustering final model.

    """

    # Clustering tags per web-page.
    clstr_y = res_h5file.get_node(params_path, name='clusters_y').read()

    # Expected class tag per web-page.
    clss_y = res_h5file.get_node(params_path, name='expected_y').read()

    # Clustering parameters for this Clustering outcome.
    clstr_params = res_h5file.get_node(params_path, name='clustering_params').read()

    if isinstance(class_tag, int):

        # Selecting the indeces for a specific Class.
        cls_tag_idxs = np.where(clss_y == class_tag)

        # Getting the Cluster tags for the respective Class documents of the specific Class.
        clstr_y = clstr_y[cls_tag_idxs]

        # Getting the Class documents of the specific Class.
        clss_y = clss_y[cls_tag_idxs]

    # Returning Cluster tags, Class tags and Clustering parameters.
    return clstr_y, clss_y, clstr_params
