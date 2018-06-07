from inception_score import get_inception_score
import numpy as np 
import tensorflow as tf
from PIL import Image 
import argparse 
import glob 
from skimage.measure import compare_ssim as ssim 
from models.util import Progbar

poster_directory = 'posters/'
sample_directory = 'output/'
subset = 1000

def get_gold(genre):
    dataset = poster_directory + genre 
    image_list = np.zeros((subset, 64, 64, 3))
    count = 0
    for filename in glob.glob(dataset + '/*.jpg'):
        if count >= 1000:
            break
        img = Image.open(filename)
        img = img.resize((64, 64))
        img = np.array(img)
        image_list[count] = img 
        count += 1 
    print("Gold", image_list.shape) 
    return image_list.astype(np.float32)

def get_samples(genre, samples): 
    dataset = sample_directory + genre + "/" + samples 
    image_list = [] 
    for filename in glob.glob(dataset + '/*.jpg'):
        img = Image.open(filename)
        img = np.array(img).astype(np.float32)
        image_list.append(img) 
    return image_list

def get_inception_scores(samples):
    score = get_inception_score(samples)
    print("Inception Mean", score[0])
    print("Inception Std", score[1])
    
def get_ssim(gold, samples):
    samples = np.array(samples)
    ssims = np.zeros((samples.shape[0], gold.shape[0]))
    pbar = Progbar(target=samples.shape[0] * gold.shape[0])
    count = 0
    for i in range(0, samples.shape[0]):
        img = samples[i] 
        for j in range(0, gold.shape[0]):
            gold_img = gold[j]
            score = ssim(img, gold_img, multichannel=True) 
            ssims[i, j] = score
            count += 1
            pbar.update(count)
    ssims_averages = np.mean(ssims, axis=0) 
    ssims_averages = np.mean(ssims_averages, axis=0)
    ssims_std = np.std(ssims)
    print("SSIM Mean:", ssims_averages)
    print("SSIM Std:", ssims_std)

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--genre', 
            type=str, 
            default='Romance')
    parser.add_argument(
            '--samples', 
            type=str, 
            default='DCGAN')

    FLAGS, _ = parser.parse_known_args()

    gold = get_gold(FLAGS.genre)
    samples = get_samples(FLAGS.genre, FLAGS.samples)

    get_inception_scores(samples)
    get_ssim(gold, samples)
    # get_inception_scores(samples)


if __name__=='__main__': 
    main()
    
