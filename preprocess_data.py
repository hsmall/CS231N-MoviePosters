import glob
import numpy as np
import re


def filter_genres(all_genres_list, used_genres_list, genres):
	indices = [ all_genres_list.index(genre) for genre in used_genres_list ]
	return genres[:, indices]


def main():
	print('Combining split up data files...')
	sort_fn = lambda x: int(re.findall('(\d*).npy', x)[0])
	poster_files = sorted(glob.glob('movie_data/posters_*.npy'), key = sort_fn)
	genre_files = sorted(glob.glob('movie_data/genres_*.npy'), key = sort_fn)
	
	print('Loading Files...')
	poster_data = [ np.load(file) for file in poster_files ]
	genre_data = [ np.load(file) for file in genre_files ]
	
	print('Concatenating Files...')
	posters = np.concatenate(poster_data, axis=0)
	genres = np.concatenate(genre_data, axis=0)
	

	print('Saving Files...')
	np.save('movie_data/posters.npy', posters, allow_pickle=False)
	np.save('movie_data/genres.npy', genres, allow_pickle=False)

	print(posters.shape, genres.shape)
	print('Done.')

if __name__ == '__main__':
	main()
