import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage 
from scipy.spatial.distance import pdist, squareform
import seaborn as sn

def main():
	genre_names = np.array([
		'Action', 'Adventure', 'Animation', 'Comedy', 
		'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy',
		'History', 'Horror', 'Music', 'Mystery', 'Romance',
		'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western'
	])
	genres = np.load('movie_data/genres.npy')
	distances = pdist(genres.T, 'jaccard')
	D = squareform(distances)
	
	clustering = linkage(distances, method='complete')
	#fig = plt.figure(figsize=(25, 10))
	#dn = dendrogram(clustering, labels=genre_names, leaf_rotation=90)
	#plt.show()

	cluster_indices = fcluster(clustering, 1.0) - 1
	num_clusters = max(cluster_indices) + 1
	print('Clusters:')
	for index in range(num_clusters):
		print(index, genre_names[np.where(cluster_indices == index)])
	print('')

	num_examples = genres.shape[0]
	clustered_genres = np.zeros((num_examples, num_clusters), dtype=np.uint8)
	count = 0
	for i in range(num_examples):
		if sum(genres[i]) == 0: count += 1
		for j in range(len(genres[i])):
			if genres[i, j] == 1: clustered_genres[i, cluster_indices[j]] = 1

	#print(np.mean(genres, axis=0))
	#print(np.mean(clustered_genres, axis=0))
	#np.save('movie_data/clustered_genres.npy', clustered_genres, allow_pickle=False)

if __name__ == '__main__':
	main()
