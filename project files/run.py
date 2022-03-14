from cmath import log
from inspect import Attribute
from operator import index
from re import L
from turtle import pos
from utils import *
import pprint
from collections import Counter
import math
import matplotlib.pyplot as plt

def count_occurances(pos_train, neg_train):
	pos_counts = {}
	pos_total_words = 0
	for doc in pos_train:
		for word in doc:
			if word not in pos_counts.keys():
				pos_counts[word] = 1
				pos_total_words += 1
			else:
				pos_counts[word] += 1
				pos_total_words += 1


	neg_counts = {}
	neg_total_words = 0
	for doc in neg_train:
		for word in doc:
			if word not in neg_counts.keys():
				neg_counts[word] = 1
				neg_total_words += 1
			else:
				neg_counts[word] += 1
				neg_total_words += 1
	
	return pos_counts, neg_counts

def calculate_prob(mode, word, counts, vocab, total_words, alpha):
	if mode == 'log and alpha':
		if word not in counts.keys():
			return math.log((alpha) / ((alpha*len(vocab)) + total_words))
		else:
			return math.log((counts[word] + alpha) / ((alpha*len(vocab)) + total_words))
	elif mode == 'alpha without log':
		if word not in counts.keys():
			return (alpha) / ((alpha*len(vocab)) + total_words)
		else:
			return (counts[word] + alpha) / ((alpha*len(vocab)) + total_words)
	elif mode == 'log without alpha':
		if word in counts.keys():
			return math.log(counts[word]/ total_words)
		else:
			return 0
	elif mode == 'without log and alpha':
		if word in counts.keys(): 
			return counts[word]/ total_words
		else:
			return 0

def execute(mode, pos_test, neg_test, pos_counts, neg_counts, pos_prior, neg_prior, vocab, alpha):

	positive_probability, negative_probability, true_positive, true_negative, false_negative, false_positive = 0, 0, 0, 0, 0, 0
	pos_total_words = sum(pos_counts.values())
	neg_total_words = sum(neg_counts.values())
	

	for doc in pos_test:
		if mode == 'without log and alpha' or mode == 'alpha without log':
			pos_prob = 1
			neg_prob = 1
			for word in doc:
				pos_prob *= calculate_prob(mode, word, pos_counts, vocab, pos_total_words, alpha)
				neg_prob *= calculate_prob(mode, word, neg_counts, vocab, neg_total_words, alpha)
			positive_probability = pos_prior * pos_prob
			negative_probability = neg_prior * neg_prob
		elif mode == 'log and alpha' or mode == 'log without alpha':
			pos_prob = 0
			neg_prob = 0
			for word in doc:
				pos_prob += calculate_prob(mode, word, pos_counts, vocab, pos_total_words, alpha)
				neg_prob += calculate_prob(mode, word, neg_counts, vocab, neg_total_words, alpha)
			positive_probability = math.log(pos_prior) + pos_prob
			negative_probability = math.log(neg_prior) + neg_prob	
		
		if(positive_probability > negative_probability):
			true_positive += 1
		elif(positive_probability < negative_probability):
			false_negative += 1
		else:
			choice = random.choice(['pos', 'neg'])
			if(choice == 'pos'):
				true_positive +=1
			elif(choice == 'neg'):
				false_negative +=1
		
	for doc in neg_test:
		if mode == 'without log and alpha' or mode == 'alpha without log':
			pos_prob = 1
			neg_prob = 1
			for word in doc:
				pos_prob *= calculate_prob(mode, word, pos_counts, vocab, pos_total_words, alpha)
				neg_prob *= calculate_prob(mode, word, neg_counts, vocab, neg_total_words, alpha)
			positive_probability = pos_prior * pos_prob
			negative_probability = neg_prior * neg_prob
		elif mode == 'log and alpha' or mode == 'log without alpha':
			pos_prob = 0
			neg_prob = 0
			for word in doc:
				pos_prob += calculate_prob(mode, word, pos_counts, vocab, pos_total_words, alpha)
				neg_prob += calculate_prob(mode, word, neg_counts, vocab, neg_total_words, alpha)
			positive_probability = math.log(pos_prior) + pos_prob
			negative_probability = math.log(neg_prior) + neg_prob	

		if(positive_probability > negative_probability):
			false_positive += 1
		elif(positive_probability < negative_probability):
			true_negative +=1
		else:
			choice = random.choice(['pos', 'neg'])
			if(choice == 'pos'):
				false_positive +=1
			elif(choice == 'neg'):
				true_negative +=1

	accuracy = (true_positive + true_negative) / (true_negative + true_positive + false_negative + false_positive)
	precision = true_positive / (true_positive + false_negative)
	recall = true_positive / (true_positive + false_positive)
	confusion_matrix = [ [true_positive, false_positive] , [false_negative, true_negative] ]

	return accuracy, precision, recall, confusion_matrix



if __name__=="__main__":

	print('hi, your code runs')

	#experiment 1

	#part 1
	mode = 'without log and alpha'
	alpha = 0
	percentage_positive_instances_train = 0.2
	percentage_negative_instances_train = 0.2
	percentage_positive_instances_test  = 0.2
	percentage_negative_instances_test  = 0.2
	(pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
	(pos_test,  neg_test)         = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)
	pos_prior = len(pos_train)/(len(pos_train) + len(neg_train))
	neg_prior = len(neg_train)/(len(pos_train) + len(neg_train))
	pos_counts, neg_counts = count_occurances(pos_train, neg_train)
	accuracy, precision, recall, confusion_matrix = execute(mode, pos_test, neg_test, pos_counts, neg_counts, pos_prior, neg_prior, vocab, alpha)


	print(mode)
	print('---------------------')
	print(accuracy)
	print(precision)
	print(recall)
	print(confusion_matrix)
	print()
	
	#part 2
	mode = 'log without alpha'
	alpha = 0
	percentage_positive_instances_train = 0.2
	percentage_negative_instances_train = 0.2
	percentage_positive_instances_test  = 0.2
	percentage_negative_instances_test  = 0.2
	(pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
	(pos_test,  neg_test)         = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)
	pos_prior = len(pos_train)/(len(pos_train) + len(neg_train))
	neg_prior = len(neg_train)/(len(pos_train) + len(neg_train))
	pos_counts, neg_counts = count_occurances(pos_train, neg_train)
	accuracy, precision, recall, confusion_matrix = execute(mode, pos_test, neg_test, pos_counts, neg_counts, pos_prior, neg_prior, vocab, alpha)
	print(mode)
	print('---------------------')
	print(accuracy)
	print(precision)
	print(recall)
	print(confusion_matrix)
	print()


	#experiment 2

	#part 1
	mode = 'log and alpha'
	alpha = 1
	percentage_positive_instances_train = 0.2
	percentage_negative_instances_train = 0.2
	percentage_positive_instances_test  = 0.2
	percentage_negative_instances_test  = 0.2
	(pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
	(pos_test,  neg_test)         = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)
	pos_prior = len(pos_train)/(len(pos_train) + len(neg_train))
	neg_prior = len(neg_train)/(len(pos_train) + len(neg_train))
	pos_counts, neg_counts = count_occurances(pos_train, neg_train)
	accuracy, precision, recall, confusion_matrix = execute(mode, pos_test, neg_test, pos_counts, neg_counts, pos_prior, neg_prior, vocab, alpha)
	print(mode)
	print('---------------------')
	print(accuracy)
	print(precision)
	print(recall)
	print(confusion_matrix)
	print()
	
	#part 2
	print('alpha analysis')
	mode = 'log and alpha'
	alpha = 0.0001
	percentage_positive_instances_train = 0.2
	percentage_negative_instances_train = 0.2
	percentage_positive_instances_test  = 0.2
	percentage_negative_instances_test  = 0.2
	(pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
	(pos_test,  neg_test)         = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)
	pos_prior = len(pos_train)/(len(pos_train) + len(neg_train))
	neg_prior = len(neg_train)/(len(pos_train) + len(neg_train))
	pos_counts, neg_counts = count_occurances(pos_train, neg_train)
	accuracies = []
	xaxis = []
	while(alpha < 1001):
		accuracy, precision, recall, confusion_matrix = execute(mode, pos_test, neg_test, pos_counts, neg_counts, pos_prior, neg_prior, vocab, alpha)
		accuracies.append(accuracy*100)
		xaxis.append(math.log10(alpha))
		alpha = alpha * 10
	print(accuracies)
	print(xaxis)
	best_alpha = math.pow(10,(xaxis[accuracies.index(max(accuracies))]))
	print('The best value of alpha is ', best_alpha)
	plt.plot(xaxis, accuracies)
	plt.show()
	print()
	best_alpha = 10


	#experiment 3
	print('best alpha on 100 percent of training data')
	mode = 'log and alpha'
	alpha = best_alpha
	percentage_positive_instances_train = 1
	percentage_negative_instances_train = 1
	percentage_positive_instances_test  = 1
	percentage_negative_instances_test  = 1
	(pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
	(pos_test,  neg_test)         = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)
	pos_prior = len(pos_train)/(len(pos_train) + len(neg_train))
	neg_prior = len(neg_train)/(len(pos_train) + len(neg_train))
	pos_counts, neg_counts = count_occurances(pos_train, neg_train)
	accuracy, precision, recall, confusion_matrix = execute(mode, pos_test, neg_test, pos_counts, neg_counts, pos_prior, neg_prior, vocab, alpha)
	print(mode)
	print('---------------------')
	print(accuracy)
	print(precision)
	print(recall)
	print(confusion_matrix)
	print()


	# experiment 4
	best_alpha = 10
	print('best alpha on 50 percent of training data')
	mode = 'log and alpha'
	alpha = best_alpha
	percentage_positive_instances_train = 0.5
	percentage_negative_instances_train = 0.5
	percentage_positive_instances_test  = 1
	percentage_negative_instances_test  = 1
	(pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
	(pos_test,  neg_test)         = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)
	pos_prior = len(pos_train)/(len(pos_train) + len(neg_train))
	neg_prior = len(neg_train)/(len(pos_train) + len(neg_train))
	pos_counts, neg_counts = count_occurances(pos_train, neg_train)
	accuracy, precision, recall, confusion_matrix = execute(mode, pos_test, neg_test, pos_counts, neg_counts, pos_prior, neg_prior, vocab, alpha)
	print(mode)
	print('---------------------')
	print(accuracy)
	print(precision)
	print(recall)
	print(confusion_matrix)
	print()

	
	
	# # experiment 6
	print('best alpha on unbalanced data')
	mode = 'log and alpha'
	alpha = best_alpha
	percentage_positive_instances_train = 0.1
	percentage_negative_instances_train = 0.5
	percentage_positive_instances_test  = 1
	percentage_negative_instances_test  = 1
	(pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
	(pos_test,  neg_test)         = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)
	pos_prior = len(pos_train)/(len(pos_train) + len(neg_train))
	neg_prior = len(neg_train)/(len(pos_train) + len(neg_train))
	pos_counts, neg_counts = count_occurances(pos_train, neg_train)
	accuracy, precision, recall, confusion_matrix = execute(mode, pos_test, neg_test, pos_counts, neg_counts, pos_prior, neg_prior, vocab, alpha)
	print(mode)
	print('---------------------')
	print(accuracy)
	print(precision)
	print(recall)
	print(confusion_matrix)
	print()


