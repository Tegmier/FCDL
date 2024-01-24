import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer,util
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics

model = SentenceTransformer('stsb-xlm-r-multilingual')
# model = SentenceTransformer('oshizo/sbert-jsnli-luke-japanese-base-lite')
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

with open('D:\\code\\FCDL\\dataleaves\\extraction.pkl', 'rb') as f:
    fc_all = pd.DataFrame(pickle.load(f))

test_sentence = pd.read_csv("D://code//FCDL//test_sentences.csv")

fc_all[['Sentence', 'User', 'Answer', 'Keyword']].to_csv('Features Concepts.csv', index = False)
fc_all['Keyword'].to_csv('Features Concept Sentences.csv', index = False)

def embedding_sentences(sentences_list):
    sentence_vector_list = []
    sentence_vector_list = model.encode(sentences_list, convert_to_tensor=True)
    return sentence_vector_list

def get_vectors(fc_all):
    vectors = []
    for i in range(len(fc_all)):
        vectors.append(np.array(fc_all["Vector"][i]))
    return vectors
    

def get_sentence_vector(in_sentence):
    return model.encode(in_sentence, convert_to_tensor=False)

def decode_sentence_vector(in_sentence):
    pass

def demonstrating_clustering_labels(clustering_model):
    labels = clustering_model.labels_
    print("----- fitted labels -----")
    print(labels)

    # print(fitted_labels)

def sentence_vector_preprocessing_before_clustering(vectors):
    norm_vectors = normalize(vectors)
    cosine_sim_vectors = cosine_similarity(vectors)
    return norm_vectors, cosine_sim_vectors

def clustering_metrics_evaluation(input_vectors, fitted_labels, model_title, preprocess_method):
    # print(fitted_labels)
    print("\n\n############# Displaying the statistical result of the " + model_title + " with preprocess method of "+ preprocess_method + " #############")
    silhousette = metrics.silhouette_score(input_vectors, fitted_labels)
    print("The Silhousette Score: ", silhousette)

    calinski = metrics.calinski_harabasz_score(input_vectors, fitted_labels)
    print("The Calinski-Harabasz Score: ", calinski)

    Davies = metrics.davies_bouldin_score(input_vectors, fitted_labels)
    print("The Davies-Bouldin Score: ", Davies)


def clustering_Kmeans(sentence_list, num_clusters, preprocess_method):
    norm_vectors, cosine_sim_vectors = sentence_vector_preprocessing_before_clustering(sentence_list)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    if preprocess_method =='normalization':
        kmeans.fit(norm_vectors)
    elif preprocess_method == 'cosine':
        kmeans.fit(cosine_sim_vectors)
    else:
        fitted_labels = []
        print("Wrong preprocess method!")
    fitted_labels = kmeans.labels_

    # Evaluation
    clustering_metrics_evaluation(norm_vectors, fitted_labels, "Kmeans_clustering", preprocess_method)
    demonstrating_clustering_labels(kmeans)
    return kmeans
    
def Agglomerative_Clustering(sentence_list, num_clusters, preprocess_method):
    norm_vectors, cosine_sim_vectors = sentence_vector_preprocessing_before_clustering(sentence_list)
    agglomerativeClustering = AgglomerativeClustering(n_clusters=num_clusters)
    if preprocess_method =='normalization':
        agglomerativeClustering.fit(norm_vectors)
    elif preprocess_method == 'cosine':
        agglomerativeClustering.fit(cosine_sim_vectors)
    else:
        fitted_labels = []
        print("Wrong preprocess method!")

    fitted_labels = agglomerativeClustering.labels_

    # Evaluation
    clustering_metrics_evaluation(norm_vectors, fitted_labels, "Agglomerative_clustering", preprocess_method)
    demonstrating_clustering_labels(agglomerativeClustering)
   



def clustering(fc_all, num_clusters):
    clusteringKeans1 = clustering_Kmeans(get_vectors(fc_all), num_clusters, "normalization")
    clusteringKeans2 = clustering_Kmeans(get_vectors(fc_all), num_clusters, "cosine")
    clusteringAgglomeractive1 = Agglomerative_Clustering(get_vectors(fc_all), num_clusters, "normalization")
    clusteringAgglomeractive2 = Agglomerative_Clustering(get_vectors(fc_all), num_clusters, "cosine")
    
# clustering(fc_all, 4)

embeded_test_sentence = embedding_sentences(test_sentence["Sentence"])

test_kmeans1 = clustering_Kmeans(embeded_test_sentence, 4, "normalization")
test_kmeans2 = clustering_Kmeans(embeded_test_sentence, 4, "cosine")
test_kmeans3 = Agglomerative_Clustering(embeded_test_sentence, 4, "normalization")
test_kmeans4 = Agglomerative_Clustering(embeded_test_sentence, 4, "cosine")

fitted_labels = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
clustering_metrics_evaluation(embeded_test_sentence, fitted_labels, "GPT", "Not Specified")

