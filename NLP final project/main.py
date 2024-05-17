import json
import numpy as np
import pandas as pd
import copy
from compare_clustering_solutions import evaluate_clustering
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
THRESHOLD = 0.6
WEIGHT = 1.4
EPOCHS = 15


def create_centroids_list(clusters):
    """
    :param clusters: dictionary of all clusters
    :return: list of centroids for each cluster respectively
    """
    centroids = list()

    for cluster_id in clusters.keys():
        centroids.append(clusters[cluster_id]["sum_of_vectors"] / len(clusters[cluster_id]["group"]))

    return centroids


def create_distances_list(request, centroids):
    """
    :param request: specific request from dataset
    :param centroids: list of all centroids of each cluster
    :return: list of distances for each cluster respectively
    """
    return np.dot(request[1], np.array(centroids).T) if centroids else []


def remove_request_from_cluster(clusters, request, index):
    """
    :param clusters: dictionary of all clusters
    :param request: specific request from dataset to remove
    :param index: the index of the specific request to remove from clusters
    :return: None
    """
    if request[2] != -1:
        clusters[request[2]]["group"].remove(index)
        clusters[request[2]]["sum_of_vectors"] -= np.array(request[1])
        if len(clusters[request[2]]["group"]) == 0:
            clusters.pop(request[2])


def create_clusters_lists(dataset, clusters, min_size):
    """
    :param dataset: list of all requests
    :param clusters: dictionary of all clusters
    :param min_size: minimum amount of requests for which a collection of requests will be defined as a valid cluster
    :return: 2 lists:
     clustered_list: list of clusters that include all requests in each cluster
     unclustered_list: list of requests that are not clustered to any cluster
    """
    clustered_list, unclustered_list = list(), list()

    for cluster in clusters.values():
        if len(cluster["group"]) >= min_size:
            clustered_list.append([dataset[i] for i in cluster["group"]])
        else:
            unclustered_list.extend([dataset[i] for i in cluster["group"]])

    return clustered_list, unclustered_list


def clustering_requests(dataset, min_size):
    """
    :param dataset: list of all requests
    :param min_size: minimum amount of requests for which a collection of requests will be defined as a valid cluster
    :return: dictionary of clusters:
             key = cluster id
             value = details of specific cluster (sum of vectors, list of requests)
    """
    cluster_id = 0
    clusters = dict()

    for epoch in range(EPOCHS):
        is_clusters_changed = False
        for i, request in enumerate(dataset):
            centroids = create_centroids_list(clusters)
            distances = create_distances_list(request, centroids)
            if len(distances) and np.max(distances) > THRESHOLD:
                index = np.argmax(distances)
                chosen_cluster_id = list(clusters.keys())[index]
                if chosen_cluster_id != request[2]:
                    remove_request_from_cluster(clusters, request, i)
                    clusters[chosen_cluster_id]["group"].append(i)
                    clusters[chosen_cluster_id]["sum_of_vectors"] += np.array(request[1])
                    dataset[i][2] = chosen_cluster_id
                    is_clusters_changed = True
            else:
                remove_request_from_cluster(clusters, request, i)
                clusters[cluster_id] = {"sum_of_vectors": np.array(copy.deepcopy(request[1])), "group": [i]}
                dataset[i][2] = cluster_id
                cluster_id += 1
                is_clusters_changed = True

        if not is_clusters_changed:
            break

    return create_clusters_lists(dataset, clusters, min_size)


def add_label_to_list(requests, cluster_labels_list, stop_words_input=None):
    """
    :param requests: specific request from dataset to remove
    :param cluster_labels_list: list of labels for each cluster respectively
    :param stop_words_input: 'english' or None
    :return: true if managed to add label to list, else except
    """
    vectorizer = CountVectorizer(ngram_range=(2, 6), stop_words=stop_words_input)

    features_data = vectorizer.fit_transform(requests)
    feature_names = vectorizer.get_feature_names_out()
    scores = features_data.sum(axis=0).A1
    names_scores_dict = dict(zip(feature_names, scores))

    # Weight function
    for ngram in names_scores_dict.keys():
        names_scores_dict[ngram] *= WEIGHT ** (len(ngram.split()) - 1)

    max_index = np.argmax(list(names_scores_dict.values()))
    label = list(names_scores_dict.keys())[max_index]
    cluster_labels_list.append(label)

    return True


def cluster_labeling(cluster_list):
    """
    :param cluster_list: list of clusters that include all requests in each cluster
    :return: list of labels for each cluster respectively
    """
    cluster_labels_list = list()

    for cluster in cluster_list:
        given_label = False
        requests = [req[0] for req in cluster]

        try:
            given_label = add_label_to_list(requests, cluster_labels_list, "english")
        except Exception as e:
            pass

        if not given_label:
            add_label_to_list(requests, cluster_labels_list)

    return cluster_labels_list


def create_cluster_dict(cluster_list, cluster_labels_list, uncluster_list):
    """
    :param cluster_list: list of clusters that include all requests in each cluster
    :param cluster_labels_list: list of labels for each cluster respectively
    :param uncluster_list: list of requests that are not clustered to any cluster
    :return: dictionary of the relevant data to write to json file (clustered and unclustered lists)
    """
    new_cluster_list = list()

    for cluster in cluster_list:
        new_cluster_list.append([req[0] for req in cluster])
    cluster_list = [{"cluster_name": label, "requests": cluster} for label, cluster in zip(cluster_labels_list, new_cluster_list)]

    uncluster_list = [req[0] for req in uncluster_list]

    return {"cluster_list": cluster_list, "unclustered": uncluster_list}


def analyze_unrecognized_requests(data_file, output_file, min_size):

    # Reading the data file
    requests = pd.read_csv(data_file)

    # Converting each request to lowercase
    requests["text"] = requests["text"].str.lower()
    requests["text"] = requests["text"].str.strip()

    # Converting the requests to embeddings
    embedding_requests = model.encode(requests["text"])

    # Creating dataset
    dataset = [[text, embedding, -1] for text, embedding in zip(requests["text"], embedding_requests)]

    # Clustering algorithm
    cluster_list, uncluster_list = clustering_requests(dataset, int(min_size))

    # Labeling the clusters
    cluster_labels_list = cluster_labeling(cluster_list)

    # Writing clusters and list of unclustered requests
    with open(output_file, "w") as fin:
        json.dump(create_cluster_dict(cluster_list, cluster_labels_list, uncluster_list), fin, indent=4)


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['min_cluster_size'])

    # todo: evaluate your clustering solution against the provided one
    # evaluate_clustering(config['example_solution_file'], config['example_solution_file'])  # invocation example
    evaluate_clustering(config['example_solution_file'], config['output_file'])
