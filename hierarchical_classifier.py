import numpy as np
from math import log
from pprint import pprint
from sklearn.preprocessing import scale, LabelEncoder, LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import log_loss
import pickle
from copy import deepcopy
import matplotlib.pyplot as plt

class model:
	'''	def __init__(self, datafile, labelfile): #use this for "old" version of data
		self.datafile = datafile
		self.labelfile = labelfile
		data = np.genfromtxt(open(datafile), delimiter=",", dtype=np.dtype(str))
		self.num_samples = int(data.shape[0])-1
		self.num_features = int(data.shape[1])-2
		self.features = data[0,1:self.num_features+1] #feature names
		self.samples = data[1:self.num_samples+1,0] #sample names
		self.x = data[1:self.num_samples+1,1:self.num_features+1].astype(np.dtype(float)) #unprocessed features
		self.x_std = scale(self.x) #features normalised to zero mean unit std dev
		labels = np.genfromtxt(open(labelfile), delimiter=",", dtype=np.dtype(str))
		labels = dict(zip(labels[1:self.num_samples+1,0],labels[1:self.num_samples+1,4:6]))
		self.labels = np.array([labels[self.samples[i][0:-3]] for i in range(self.num_samples)]) #unprocessed labels
		self.num_outputs = int(self.labels.shape[1])
		self.enc = [LabelEncoder() for i in range(self.num_outputs)]
		self.y = np.transpose(np.vstack([self.enc[i].fit_transform(self.labels[:,i]) for i in range(self.num_outputs)])) #output labels in a categorical tuple format
		self.outputs = [len(self.enc[i].classes_) for i in range(self.num_outputs)]
		self.bin = [LabelBinarizer() for i in range(self.num_outputs)]
		self.y_std = np.hstack([self.bin[i].fit_transform(self.enc[i].transform(self.labels[:,i])) for i in range(self.num_outputs)]) #output labels in a boolean tuple format'''
	
	def __init__(self, datafile, num_outputs=2, sample_name_exists=True): #use this for "new" version of data
		self.datafile = datafile
		data = np.genfromtxt(open(datafile), delimiter=",", dtype=np.dtype(str))
		num_cols = num_outputs+int(sample_name_exists)
		self.num_samples = int(data.shape[0])-1
		self.num_features = int(data.shape[1])-num_cols
		self.features = data[0,num_cols:] #feature names
		if sample_name_exists: self.samples = data[1:,0] #sample names		
		self.x = data[1:,num_cols:].astype(np.dtype(float))
		self.x_std = data[1:,num_cols:].astype(np.dtype(float)) #features already normalised to zero mean unit std dev
		self.labels = data[1:,int(sample_name_exists):num_cols] #unprocessed labels
		self.num_outputs = num_outputs
		self.enc = [LabelEncoder() for i in range(self.num_outputs)]
		self.y = np.transpose(np.vstack([self.enc[i].fit_transform(self.labels[:,i]) for i in range(self.num_outputs)])) #output labels in a categorical tuple format
		self.outputs = [len(self.enc[i].classes_) for i in range(self.num_outputs)]
		self.bin = [LabelBinarizer() for i in range(self.num_outputs)]
		self.y_std = np.hstack([self.bin[i].fit_transform(self.enc[i].transform(self.labels[:,i])) for i in range(self.num_outputs)]) #output labels in a boolean tuple format

	def save(self):
		pickle.dump(self, open(self.datafile+'.pickle','wb'))

	def shuffle(self, state=None):
		if state is None: state = np.random.get_state()
		else: np.random.set_state(state)
		np.random.shuffle(self.samples)
		np.random.set_state(state)
		np.random.shuffle(self.x)
		np.random.set_state(state)
		np.random.shuffle(self.x_std)
		np.random.set_state(state)
		np.random.shuffle(self.y)
		np.random.set_state(state)
		np.random.shuffle(self.y_std)

#class unitary_rf:
#	def __init__(self, model=None, n_estimators=100,)

def hard_scorer(estimator, X, y, per_output=False): #implements an accuracy score strategy for multioutput-multiclass y
	y_pred = estimator.predict(X)
	if y.ndim==1:
		y = y[:,None]
		y_pred = y_pred[:,None]
	check = y==y_pred
	num_samples = float(X.shape[0])
	output_scores = sum(check)/num_samples
	total_score = sum(check.all(1))/num_samples
	if per_output: return (total_score, output_scores)
	else: return total_score

def soft_scorer(estimator, X, y, per_output=False): #implements a log-loss score strategy for multioutput-multiclass y
	if y.ndim==1: y = y[:,None]
	y_pred = estimator.predict_proba(X)
	if type(y_pred) is not list:
		y_pred = [y_pred]
		classes = [estimator.n_classes_]
	else: classes = estimator.n_classes_
	labs = [[i for i in range(n)] for n in classes]
	output_scores = [-log_loss(y[:,i], y_pred[i], labels=labs[i]) for i in range(int(y.shape[1]))]
	total_score = sum(output_scores)
	if per_output: return (total_score, output_scores)
	else: return total_score

def confusion_matrix(y_test, y_pred, labels):
	cm = np.zeros([len(labels), len(labels)], dtype=np.dtype(int))
	for i in range(len(y_test)): cm[int(y_test[i])][int(y_pred[i])] += 1
	return cm

class parallel_rf:
	def __init__(self, n_classes=[], n_estimators=100, max_depth=10000, class_weight=None):
		self.n_classes_ = n_classes
		self.breadth = len(self.n_classes_)
		self.estimators = n_estimators
		self.rf_depth = max_depth
		self.class_wt = class_weight
		self.clfs = [RandomForestClassifier(n_estimators=self.estimators, max_depth=self.rf_depth, class_weight=self.class_wt) for i in range(self.breadth)]

	def fit(self, X, y, **kwargs):
		if 'n_classes' in kwargs:
			self.n_classes_ = kwargs['n_classes']
			self.breadth = len(self.n_classes_)
		if 'n_estimators' in kwargs: self.estimators = kwargs['n_estimators']
		if 'max_depth' in kwargs: self.rf_depth = kwargs['max_depth']
		if 'class_weight' in kwargs: self.class_wt = kwargs['class_weight']
		if len(kwargs)>0: self.clfs = [RandomForestClassifier(n_estimators=self.estimators, max_depth=self.rf_depth, class_weight=self.class_wt) for i in range(self.breadth)]
		[self.clfs[i].fit(X, y[:,i]) for i in range(self.breadth)]
		return self

	def predict(self, X):
		return np.transpose([self.clfs[i].predict(X) for i in range(self.breadth)])

	def predict_proba(self, X):
		return [self.clfs[i].predict_proba(X) for i in range(self.breadth)]

	def feature_importances(self):
		return [self.clfs[i].feature_importances_ for i in range(self.breadth)]

	def get_params(self, deep=False):
		return dict()

class hierarchical_rf:
	def __init__(self, n_classes=[], ordering=[], n_estimators=100, max_depth=10000, class_weight=None): #labels would contain the number of classes in each output, ordered in a top-to-bottom order of output in hierarchy, such that classes are in range [0, label)
		self.n_classes_ = n_classes
		self.depth = len(self.n_classes_)
		self.ordering = ordering
		self.estimators = n_estimators
		self.rf_depth = max_depth
		self.class_wt = class_weight
		self.clfs = [RandomForestClassifier(n_estimators=self.estimators, max_depth=self.rf_depth, class_weight=self.class_wt)]
		for i in range(self.depth-2,-1,-1):
			self.clfs = [RandomForestClassifier(n_estimators=self.estimators, max_depth=self.rf_depth, class_weight=self.class_wt), [deepcopy(self.clfs) for j in range(self.n_classes_[ordering[i]])]]

	def fit(self, X, y, depth=0, clf=None, weights=None, **kwargs): #outputs in y should be ordered the same way as in the hierarchy, and y should be in boolean tuple format
		if 'n_classes' in kwargs:
			self.n_classes_ = kwargs['n_classes']
			self.depth = len(self.n_classes_)
		if 'ordering' in kwargs: self.ordering = kwargs['ordering']
		if 'n_estimators' in kwargs: self.estimators = kwargs['n_estimators']
		if 'max_depth' in kwargs: self.rf_depth = kwargs['max_depth']
		if 'class_weight' in kwargs: self.class_wt = kwargs['class_weight']
		if 'weighted' in kwargs:
			if kwargs['weighted'] is True: weights = np.ones(int(X.shape[0]))
		if len(kwargs)>0:
			self.clfs = [RandomForestClassifier(n_estimators=self.estimators, max_depth=self.rf_depth, class_weight=self.class_wt)]
			for i in range(self.depth-2,-1,-1):
				self.clfs = [RandomForestClassifier(n_estimators=self.estimators, max_depth=self.rf_depth, class_weight=self.class_wt), [deepcopy(self.clfs) for j in range(self.n_classes_[self.ordering[i]])]]
		print 'Fitting depth', depth
		if int(X.shape[0])==0: return self #nothing to be fit
		if clf==None: clf = self.clfs
		clf[0].fit(X, y[:,self.ordering[depth]], weights) #fitting data to current classifier
		if depth<self.depth-1: #depth-first exploration of subtree of current classifier
			if weights is None: [self.fit(X[y[:,self.ordering[depth]]==i,:], y[y[:,self.ordering[depth]]==i,:], depth+1, clf[1][i]) for i in range(self.n_classes_[self.ordering[depth]])]
			else: #soft-training method
				weights = np.transpose(weights*np.transpose(clf[0].predict_proba(X)))
				weights = weights/sum(weights)
				[self.fit(X, y, depth+1, clf[1][i], weights[:,i]) for i in range(self.n_classes_[self.ordering[depth]])]
		return self

	def predict(self, X, depth=0, clf=None):
		if clf==None: clf = self.clfs
		y = clf[0].predict(X)
		if depth<self.depth-1: #non-leaf, exploration of apt child of current classifier
			if depth==0:
				if self.ordering[depth+1]>self.ordering[depth]:	y = np.transpose([y, np.hstack([self.predict(X[i,:][None,:], depth+1, clf[1][y[i]]) for i in range(len(y))])])
				else: y = np.transpose([np.hstack([self.predict(X[i,:][None,:], depth+1, clf[1][y[i]]) for i in range(len(y))]), y])
			else:
				if self.ordering[depth+1]>self.ordering[depth]:	y = np.hstack([y, self.predict(X, depth+1, clf[1][y])])
				else: y = np.hstack([self.predict(X, depth+1, clf[1][y]), y])
		return y

	def predict_proba(self, X, depth=0, clf=None):
		if clf==None: clf=self.clfs
		y = clf[0].predict_proba(X)
		if depth<self.depth-1: #non-leaf, exploration of apt child of current classifier
			if self.ordering[depth+1]>self.ordering[depth]: y = [y] + reduce(lambda item1, item2: [item1[i]+item2[i] for i in range(len(item1))], map(lambda item: [np.transpose(item[0]*np.transpose(prob)) for prob in item[1]], [(y[:,i], self.predict_proba(X, depth+1, clf[1][i])) for i in range(int(y.shape[1]))]))
			else: y = reduce(lambda item1, item2: [item1[i]+item2[i] for i in range(len(item1))], map(lambda item: [np.transpose(item[0]*np.transpose(prob)) for prob in item[1]], [(y[:,i], self.predict_proba(X, depth+1, clf[1][i])) for i in range(int(y.shape[1]))])) + [y]
			return y
		return [y]

	def feature_importances(self, depth=0, clf=None):
		if clf==None: clf=self.clfs
		y = clf[0].feature_importances_
		if depth<self.depth-1: #non-leaf, exploration of apt child of current classifier
			return [y, [self.feature_importances(depth+1, clf[1][i]) for i in range(self.n_classes_[self.ordering[depth]])]]
		return [y]

	def get_params(self, deep=False):
		return dict()

class rf_tests:
	def __init__(self, modelname, class_weight=None, estimators=100, depth=100000, k=5):
		self.modelname = modelname
		self.model = pickle.load(open(modelname))
		self.model.shuffle()
		self.estimators = estimators
		self.depth = depth
		self.class_wt = class_weight
		self.k = k

	def score_tolerance_only(self, soft=False):
		print 'Running tolerance_only'
		clf = RandomForestClassifier(n_estimators=self.estimators, max_depth=self.depth, class_weight=self.class_wt)
		scores = cross_val_score(clf, self.model.x_std, self.model.y[:,0], cv=self.k, scoring=soft_scorer if soft else hard_scorer)
		return scores

	def score_pathogen_only(self, soft=False):
		print 'Running pathogen_only'
		clf = RandomForestClassifier(n_estimators=self.estimators, max_depth=self.depth, class_weight=self.class_wt)
		scores = cross_val_score(clf, self.model.x_std, self.model.y[:,1], cv=self.k, scoring=soft_scorer if soft else hard_scorer)
		return scores

	def score_both_together(self, soft=False):
		print 'Running both_together'
		clf = RandomForestClassifier(n_estimators=self.estimators, max_depth=self.depth, class_weight=self.class_wt)
		scores = cross_val_score(clf, self.model.x_std, self.model.y, cv=self.k, scoring=soft_scorer if soft else hard_scorer)
		return scores

	def score_both_separately(self, soft=False):
		print 'Running both_separately'
		params = {'n_classes':self.model.outputs, 'n_estimators':self.estimators, 'max_depth':self.depth, 'class_weight':self.class_wt}
		clf = parallel_rf(self.model.outputs, self.estimators, self.depth, self.class_wt)
		scores = cross_val_score(clf, self.model.x_std, self.model.y, cv=self.k, scoring=soft_scorer if soft else hard_scorer, fit_params=params)
		return scores

	def score_pathogen_given_tolerance(self, soft=False, weighted=False):
		print 'Running pathogen_given_tolerance'
		params = {'n_classes':self.model.outputs, 'ordering':[0,1], 'n_estimators':self.estimators, 'max_depth':self.depth, 'class_weight':self.class_wt, 'weighted':weighted}
		clf = hierarchical_rf(self.model.outputs, [0,1], self.estimators, self.depth, self.class_wt)
		scores = cross_val_score(clf, self.model.x_std, self.model.y, cv=self.k, scoring=soft_scorer if soft else hard_scorer, fit_params=params)
		return scores

	def score_tolerance_given_pathogen(self, soft=False, weighted=False):
		print 'Running tolerance_given_pathogen'
		params = {'n_classes':self.model.outputs, 'ordering':[1,0], 'n_estimators':self.estimators, 'max_depth':self.depth, 'class_weight':self.class_wt, 'weighted':weighted}
		clf = hierarchical_rf(self.model.outputs, [1,0], self.estimators, self.depth, self.class_wt)
		scores = cross_val_score(clf, self.model.x_std, self.model.y, cv=self.k, scoring=soft_scorer if soft else hard_scorer, fit_params=params)
		return scores

	def confusion_tolerance_only(self, X_train, X_test, y_train, y_test):
		print 'Running tolerance_only'
		clf = RandomForestClassifier(n_estimators=self.estimators, max_depth=self.depth, class_weight=self.class_wt)
		clf.fit(X_train, y_train[:,0])
		y_pred = clf.predict(X_test)
		return confusion_matrix(y_test[:,0], y_pred, labels=[0,1])

	def confusion_pathogen_only(self, X_train, X_test, y_train, y_test):
		print 'Running pathogen_only'
		clf = RandomForestClassifier(n_estimators=self.estimators, max_depth=self.depth, class_weight=self.class_wt)
		clf.fit(X_train, y_train[:,1])
		y_pred = clf.predict(X_test)
		return confusion_matrix(y_test[:,1], y_pred, labels=[0,1,2,3])

	def confusion_both_together(self, X_train, X_test, y_train, y_test):
		print 'Running both_together'
		clf = RandomForestClassifier(n_estimators=self.estimators, max_depth=self.depth, class_weight=self.class_wt)
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		labels = [[j for j in range(self.model.outputs[i])] for i in range(self.model.num_outputs)]
		return [confusion_matrix(y_test[:,i], y_pred[:,i], labels=labels[i]) for i in range(int(y_train.shape[1]))]

	def confusion_both_separately(self, X_train, X_test, y_train, y_test):
		print 'Running both_separately'
		clf = parallel_rf(self.model.outputs, self.estimators, self.depth, self.class_wt)
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		labels = [[j for j in range(self.model.outputs[i])] for i in range(self.model.num_outputs)]
		return [confusion_matrix(y_test[:,i], y_pred[:,i], labels=labels[i]) for i in range(int(y_train.shape[1]))]

	def confusion_pathogen_given_tolerance(self, X_train, X_test, y_train, y_test, weighted=False):
		print 'Running pathogen_given_tolerance'
		clf = hierarchical_rf(self.model.outputs, [0,1], self.estimators, self.depth, self.class_wt)
		clf.fit(X_train, y_train, weighted=weighted)
		y_pred = clf.predict(X_test)
		ordering = [0,1]
		labels = [[j for j in range(self.model.outputs[i])] for i in range(self.model.num_outputs)]
		cms = [[[[] for b in range(self.model.outputs[ordering[i]])] for a in range(self.model.outputs[ordering[i]])] for i in range(int(y_train.shape[1])-1)]
		for i in range(int(y_train.shape[1])-1):
			for a in range(self.model.outputs[ordering[i]]):
				for b in range(self.model.outputs[ordering[i]]):
					a_t = y_test[:, ordering[i]]==a
					b_t = y_pred[:, ordering[i]]==b
					cms[i][a][b] = confusion_matrix(y_test[a_t&b_t, ordering[i+1]], y_pred[a_t&b_t, ordering[i+1]], labels=labels[ordering[i+1]])
		return [[confusion_matrix(y_test[:,i], y_pred[:,i], labels=labels[i]) for i in range(int(y_train.shape[1]))], cms]

	def confusion_tolerance_given_pathogen(self, X_train, X_test, y_train, y_test, weighted=False):
		print 'Running tolerance_given_pathogen'
		clf = hierarchical_rf(self.model.outputs, [1,0], self.estimators, self.depth, self.class_wt)
		clf.fit(X_train, y_train, weighted=weighted)
		y_pred = clf.predict(X_test)
		ordering = [1,0]
		labels = [[j for j in range(self.model.outputs[i])] for i in range(self.model.num_outputs)]
		cms = [[[[] for b in range(self.model.outputs[ordering[i]])] for a in range(self.model.outputs[ordering[i]])] for i in range(int(y_train.shape[1])-1)]
		for i in range(int(y_train.shape[1])-1):
			for a in range(self.model.outputs[ordering[i]]):
				for b in range(self.model.outputs[ordering[i]]):
					a_t = y_test[:, ordering[i]]==a
					b_t = y_pred[:, ordering[i]]==b
					cms[i][a][b] = confusion_matrix(y_test[a_t&b_t, ordering[i+1]], y_pred[a_t&b_t, ordering[i+1]], labels=labels[ordering[i+1]])
		return [[confusion_matrix(y_test[:,i], y_pred[:,i], labels=labels[i]) for i in range(int(y_train.shape[1]))], cms]
	
	def features_tolerance_only(self):
		print 'Running tolerance_only'
		clf = RandomForestClassifier(n_estimators=self.estimators, max_depth=self.depth, class_weight=self.class_wt)
		clf.fit(self.model.x_std, self.model.y[:,0])
		return clf.feature_importances_

	def features_pathogen_only(self):
		print 'Running pathogen_only'
		clf = RandomForestClassifier(n_estimators=self.estimators, max_depth=self.depth, class_weight=self.class_wt)
		clf.fit(self.model.x_std, self.model.y[:,1])
		return clf.feature_importances_

	def features_both_together(self):
		print 'Running both_together'
		clf = RandomForestClassifier(n_estimators=self.estimators, max_depth=self.depth, class_weight=self.class_wt)
		clf.fit(self.model.x_std, self.model.y)
		return clf.feature_importances_

	def features_both_separately(self):
		print 'Running both_separately'
		clf = parallel_rf(self.model.outputs, self.estimators, self.depth, self.class_wt)
		clf.fit(self.model.x_std, self.model.y)
		return clf.feature_importances()

	def features_pathogen_given_tolerance(self, weighted=False):
		print 'Running pathogen_given_tolerance'
		clf = hierarchical_rf(self.model.outputs, [0,1], self.estimators, self.depth, self.class_wt)
		clf.fit(self.model.x_std, self.model.y, weighted=weighted)
		return clf.feature_importances()

	def features_tolerance_given_pathogen(self, weighted=False):
		print 'Running tolerance_given_pathogen'
		clf = hierarchical_rf(self.model.outputs, [1,0], self.estimators, self.depth, self.class_wt)
		clf.fit(self.model.x_std, self.model.y, weighted=weighted)
		return clf.feature_importances()

	def score_all(self, scorefile=None, save=False):
		if scorefile is None or save:
			scores = [[],[]]
			scores[0].append(self.score_tolerance_only())
			scores[0].append(self.score_pathogen_only())
			scores[0].append(self.score_both_together())
			scores[0].append(self.score_both_separately())
			scores[0].append(self.score_pathogen_given_tolerance())
			scores[0].append(self.score_pathogen_given_tolerance(False, True))
			scores[0].append(self.score_tolerance_given_pathogen())
			scores[0].append(self.score_tolerance_given_pathogen(False, True))
			scores[1].append(self.score_tolerance_only(True))
			scores[1].append(self.score_pathogen_only(True))
			scores[1].append(self.score_both_together(True))
			scores[1].append(self.score_both_separately(True))
			scores[1].append(self.score_pathogen_given_tolerance(True))
			scores[1].append(self.score_pathogen_given_tolerance(True, True))
			scores[1].append(self.score_tolerance_given_pathogen(True))
			scores[1].append(self.score_tolerance_given_pathogen(True, True))
		if scorefile is not None:
			if save: pickle.dump(scores, open(scorefile,'wb'))
			else: scores = pickle.load(open(scorefile))
		x = np.arange(8)
		width = 0.35
		fig, ax = plt.subplots()
		colors = ['r','g']
		plts = [ax.bar(x+width*i, np.array(scores[i]).mean(1), width, color=colors[i], yerr=np.array(scores[i]).std(1)) for i in range(2)]
		ax.set_title('Scores for various RF Algorithms')
		ax.set_ylabel('Score')
		ax.set_xticks(x+width)
		ax.set_xticklabels(('tolerance_only', 'pathogen_only', 'both_together', 'both_separately', 'pathogen_given_tolerance', 'pathogen_given_tolerance_soft', 'tolerance_given_pathogen', 'tolerance_given_pathogen_soft'))
		ax.legend((plts[0][0], plts[1][0]), ('accuracy','negative log loss'), loc=3)
		[[ax.text(rect.get_x() + rect.get_width()/2., -0.05+rect.get_y() if rect.get_y()<0 else 0.025+rect.get_height(), '%.3f' % round(rect.get_y(), 3) if rect.get_y()<0 else round(rect.get_height(), 3), ha='center', va='bottom') for rect in rects] for rects in plts]
		plt.show()

	def plot_confusion_matrix(self, cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, text=True):
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		ticks = np.arange(len(classes))
		plt.xticks(ticks, classes, rotation=45)
		plt.yticks(ticks, classes)
		thresh = cm.max()/2.
		[[plt.text(j, i, cm[i,j], horizontalalignment="center", color="white" if cm[i,j]>thresh else "black") for j in range(len(classes))] for i in range(len(classes))]
		if text:			
			plt.title(title)
			plt.xlabel('Predicted Label')
			plt.ylabel('True Label')

	def confusion_all(self, scorefile=None, save=False, n=5):
		if scorefile is None or save:
			scores = [[] for i in range(8)]
			for i in range(n):
				X_train, X_test, y_train, y_test = train_test_split(self.model.x_std, self.model.y)
				scores[0].append(self.confusion_tolerance_only(X_train, X_test, y_train, y_test))
				scores[1].append(self.confusion_pathogen_only(X_train, X_test, y_train, y_test))
				scores[2].append(self.confusion_both_together(X_train, X_test, y_train, y_test))
				scores[3].append(self.confusion_both_separately(X_train, X_test, y_train, y_test))
				scores[4].append(self.confusion_pathogen_given_tolerance(X_train, X_test, y_train, y_test))
				scores[5].append(self.confusion_pathogen_given_tolerance(X_train, X_test, y_train, y_test, True))
				scores[6].append(self.confusion_tolerance_given_pathogen(X_train, X_test, y_train, y_test))
				scores[7].append(self.confusion_tolerance_given_pathogen(X_train, X_test, y_train, y_test, True))
		if scorefile is not None:
			if save: pickle.dump(scores, open(scorefile,'wb'))
			else: scores = pickle.load(open(scorefile))
		labels = ('tolerance_only', 'pathogen_only', 'both_together', 'both_separately', 'pathogen_given_tolerance', 'pathogen_given_tolerance_soft', 'tolerance_given_pathogen', 'tolerance_given_pathogen_soft')
		plt.figure()
		self.plot_confusion_matrix(reduce(lambda item1, item2: np.add(item1, item2), scores[0]), self.model.enc[0].classes_, 'Tolerance Confusion Matrix '+labels[0], plt.cm.Reds)
		plt.show()
		plt.clf()
		plt.figure()
		self.plot_confusion_matrix(reduce(lambda item1, item2: np.add(item1, item2), scores[1]), self.model.enc[1].classes_, 'Pathogen Confusion Matrix '+labels[1], plt.cm.Blues)
		plt.show()
		plt.clf()
		for i in range(2, 4):
			plt.figure()
			plt.subplot(1, 2, 1)
			self.plot_confusion_matrix(reduce(lambda item1, item2: np.add(item1, item2), [score[0] for score in scores[i]]), self.model.enc[0].classes_, 'Tolerance Confusion Matrix '+labels[i], plt.cm.Reds)
			plt.subplot(1, 2, 2)
			self.plot_confusion_matrix(reduce(lambda item1, item2: np.add(item1, item2), [score[1] for score in scores[i]]), self.model.enc[1].classes_, 'Pathogen Confusion Matrix '+labels[i], plt.cm.Blues)
			plt.show()
			plt.clf()
		for i in range(4, 6):
			plt.figure()
			plt.subplot(1, 2, 1)
			self.plot_confusion_matrix(reduce(lambda item1, item2: np.add(item1, item2), [score[0][0] for score in scores[i]]), self.model.enc[0].classes_, 'Tolerance Confusion Matrix '+labels[i], plt.cm.Reds)
			plt.subplot(1, 2, 2)
			self.plot_confusion_matrix(reduce(lambda item1, item2: np.add(item1, item2), [score[0][1] for score in scores[i]]), self.model.enc[1].classes_, 'Pathogen Confusion Matrix '+labels[i], plt.cm.Blues)
			plt.show()
			plt.clf()
			plt.figure()
			plt.subplot(1, 2, 1)
			self.plot_confusion_matrix(reduce(lambda item1, item2: np.add(item1, item2), [score[0][0] for score in scores[i]]), self.model.enc[0].classes_, 'Tolerance Confusion Matrix '+labels[i], plt.cm.Reds)
			grid = [[3,4],[7,8]]
			for j in range(2):
				for k in range(2):
					plt.subplot(2, 4, grid[j][k])
					self.plot_confusion_matrix(reduce(lambda item1, item2: np.add(item1, item2), [score[1][0][j][k] for score in scores[i]]), self.model.enc[1].classes_, None, plt.cm.Blues, False)
			plt.show()
			plt.clf()
		for i in range(6, 8):
			plt.figure()
			plt.subplot(1, 2, 1)
			self.plot_confusion_matrix(reduce(lambda item1, item2: np.add(item1, item2), [score[0][0] for score in scores[i]]), self.model.enc[0].classes_, 'Tolerance Confusion Matrix '+labels[i], plt.cm.Reds)
			plt.subplot(1, 2, 2)
			self.plot_confusion_matrix(reduce(lambda item1, item2: np.add(item1, item2), [score[0][1] for score in scores[i]]), self.model.enc[1].classes_, 'Pathogen Confusion Matrix '+labels[i], plt.cm.Blues)
			plt.show()
			plt.figure()
			plt.subplot(1, 2, 1)
			self.plot_confusion_matrix(reduce(lambda item1, item2: np.add(item1, item2), [score[0][1] for score in scores[i]]), self.model.enc[1].classes_, 'Pathogen Confusion Matrix '+labels[i], plt.cm.Blues)
			grid = [[5,6,7,8],[13,14,15,16],[21,22,23,24],[29,30,31,32]]
			for j in range(4):
				for k in range(4):
					plt.subplot(4, 8, grid[j][k])
					self.plot_confusion_matrix(reduce(lambda item1, item2: np.add(item1, item2), [score[1][0][j][k] for score in scores[i]]), self.model.enc[0].classes_, None, plt.cm.Reds, False)
			plt.show()
			plt.clf()

	def key_features(self, outfile=None, featurefile=None, save=False, k=2000):
		if featurefile is None or save:
			feature_importances = []
			feature_importances.append(self.features_tolerance_only())
			feature_importances.append(self.features_pathogen_only())
			feature_importances.append(self.features_both_together())
			feature_importances.append(self.features_both_separately())
			feature_importances.append(self.features_pathogen_given_tolerance())
			feature_importances.append(self.features_pathogen_given_tolerance(True))
			feature_importances.append(self.features_tolerance_given_pathogen())
			feature_importances.append(self.features_tolerance_given_pathogen(True))
		if featurefile is not None:
			if save: pickle.dump(feature_importances, open(featurefile,'wb'))
			else: feature_importances = pickle.load(open(featurefile))
		#analysing only tolerance_given_pathogen_soft
		tolerance_only_feats = map(lambda x: self.model.features[x[0]], sorted(zip(np.arange(self.model.num_features), feature_importances[0]), key=lambda x: x[1], reverse=True))
		tolerance_only_feats = set(tolerance_only_feats[:k])
		pathogen_only_feats = map(lambda x: self.model.features[x[0]], sorted(zip(np.arange(self.model.num_features), feature_importances[1]), key=lambda x: x[1], reverse=True))
		pathogen_only_feats = set(pathogen_only_feats[:k])		
		common_pathogen_only_tolerance_only_feats = tolerance_only_feats & pathogen_only_feats
		pathogen_feats = map(lambda x: self.model.features[x[0]], sorted(zip(np.arange(self.model.num_features), feature_importances[-1][0]), key=lambda x: x[1], reverse=True))
		pathogen_feats = set(pathogen_feats[:k])
		tolerance_given_pathogen_feats = [map(lambda x: self.model.features[x[0]], sorted(zip(np.arange(self.model.num_features), feat_imp[0]), key=lambda x: x[1], reverse=True)) for feat_imp in feature_importances[-1][1]]
		tolerance_given_pathogen_feats = [set(tolerance_feat[:k]) for tolerance_feat in tolerance_given_pathogen_feats]
		common_tolerance_given_pathogen_feats = reduce(lambda x, y: x & y, tolerance_given_pathogen_feats)
		common_pathogen_tolerance_given_pathogen_feats = map(lambda x: x & pathogen_feats, tolerance_given_pathogen_feats)
		common_pathogen_alltolerances_given_pathogen_feats = pathogen_feats & common_tolerance_given_pathogen_feats
		common_tolerance_only_alltolerances_given_pathogen_feats = tolerance_only_feats & common_tolerance_given_pathogen_feats
		text = 'common_pathogen_only_tolerance_only_feats ' + str(len(common_pathogen_only_tolerance_only_feats)) + '\n' + '\n'.join([i for i in list(common_pathogen_only_tolerance_only_feats)]) + '\n*\t*\t*\n\n'
		text += 'common_tolerance_given_pathogen_feats ' + str(len(common_tolerance_given_pathogen_feats)) + '\n' + '\n'.join([i for i in list(common_tolerance_given_pathogen_feats)]) + '\n*\t*\t*\n\n'
		text += 'common_pathogen_tolerance_given_pathogen_feats ' + ','.join([str(len(common_pathogen_tolerance_given_pathogen_feat)) for common_pathogen_tolerance_given_pathogen_feat in common_pathogen_tolerance_given_pathogen_feats]) + '\n' + '\n-\t-\t-\n'.join(['\n'.join([i for i in list(common_pathogen_tolerance_given_pathogen_feat)]) for common_pathogen_tolerance_given_pathogen_feat in common_pathogen_tolerance_given_pathogen_feats]) + '\n*\t*\t*\n'
		text += 'common_pathogen_alltolerances_given_pathogen_feats ' + str(len(common_pathogen_alltolerances_given_pathogen_feats)) + '\n' + '\n'.join([i for i in list(common_pathogen_alltolerances_given_pathogen_feats)]) + '\n*\t*\t*\n\n'
		text += 'common_tolerance_only_alltolerances_given_pathogen_feats ' + str(len(common_tolerance_only_alltolerances_given_pathogen_feats)) + '\n' + '\n'.join([i for i in list(common_tolerance_only_alltolerances_given_pathogen_feats)])
		with open(outfile, 'wb') as fd: fd.write(text)

	def rank_features(self, outfile=None, featurefile=None, save=False):
		if featurefile is None or save:
			feature_importances = []
			feature_importances.append(self.features_tolerance_only())
			feature_importances.append(self.features_pathogen_only())
			feature_importances.append(self.features_both_together())
			feature_importances.append(self.features_both_separately())
			feature_importances.append(self.features_pathogen_given_tolerance())
			feature_importances.append(self.features_pathogen_given_tolerance(True))
			feature_importances.append(self.features_tolerance_given_pathogen())
			feature_importances.append(self.features_tolerance_given_pathogen(True))
		if featurefile is not None:
			if save: pickle.dump(feature_importances, open(featurefile,'wb'))
			else: feature_importances = pickle.load(open(featurefile))
		pathogen_feat_ranks = np.array(feature_importances[-1][0])
		pathogen_feat_borda_ranks = np.array(map(lambda x: x[0], sorted(zip(np.arange(self.model.num_features), sorted(zip(np.arange(self.model.num_features), pathogen_feat_ranks), key=lambda x: x[1])), key=lambda x:x[1][0])))
		tolerance_given_pathogen_feat_ranks = [np.array(feat_imp[0]) for feat_imp in feature_importances[-1][1]]
		tolerance_given_pathogen_feat_borda_ranks = [np.array(map(lambda x: x[0], sorted(zip(np.arange(self.model.num_features), sorted(zip(np.arange(self.model.num_features), y), key=lambda x: x[1])), key=lambda x:x[1][0]))) for y in tolerance_given_pathogen_feat_ranks]
		tolerance_given_pathogen_feat_ranks_combo = reduce(lambda x, y: x * y, tolerance_given_pathogen_feat_ranks)
		tolerance_given_pathogen_feat_borda_ranks_combo = reduce(lambda x, y: x + y, tolerance_given_pathogen_feat_borda_ranks)
		tolerance_given_pathogen_feat_ranks_combo_all = tolerance_given_pathogen_feat_ranks_combo + pathogen_feat_ranks
		tolerance_given_pathogen_feat_borda_ranks_combo_all = tolerance_given_pathogen_feat_borda_ranks_combo + pathogen_feat_borda_ranks
		copeland_win_matrix = np.array([[sum([y[i]>y[j] for y in tolerance_given_pathogen_feat_ranks]) for j in range(self.model.num_features)] for i in range(self.model.num_features)])
		tolerance_given_pathogen_feat_copeland_ranks_combo = copeland_win_matrix.sum(1) - copeland_win_matrix.sum(0)
		copeland_win_matrix += np.array([[pathogen_feat_ranks[i]>pathogen_feat_ranks[j] for j in range(self.model.num_features)] for i in range(self.model.num_features)])
		tolerance_given_pathogen_feat_copeland_ranks_combo_all = copeland_win_matrix.sum(1) - copeland_win_matrix.sum(0)
		col_headings = ','.join(['feat_id'] + list(self.model.enc[1].classes_) + ['combo', 'combo_all', 'combo_borda', 'combo_all_borda', 'combo_copeland', 'combo_all_copeland'])
		text = '\n'.join([col_headings]+[','.join([self.model.features[i]] + [str(x[i]) for x in tolerance_given_pathogen_feat_ranks] + [str(tolerance_given_pathogen_feat_ranks_combo[i]), str(tolerance_given_pathogen_feat_ranks_combo_all[i]), str(tolerance_given_pathogen_feat_borda_ranks_combo[i]), str(tolerance_given_pathogen_feat_borda_ranks_combo_all[i]), str(tolerance_given_pathogen_feat_copeland_ranks_combo[i]), str(tolerance_given_pathogen_feat_copeland_ranks_combo_all[i])]) for i in range(self.model.num_features)])
		with open(outfile, 'wb') as fd: fd.write(text)

datafile = '../data/retrospective/influenza/csvdata.csv'
labelfile = '../data/retrospective/influenza/csvlabels.csv'
modelfile = '../data/retrospective/influenza/csvdata.csv.pickle'

stddatafile = '../data/retrospective/influenza/gsym.csv'
stdmodelfile = '../data/retrospective/influenza/gsym.csv.pickle'

scorefile = '../data/retrospective/influenza/scores.pickle'
confusionfile = '../data/retrospective/influenza/confusions.pickle'
featurefile = '../data/retrospective/influenza/features.pickle'

stdscorefile = '../data/retrospective/influenza/gsym_scores.pickle'
stdfeaturefile = '../data/retrospective/influenza/gsym_features.pickle'

scorefile_balanced = '../data/retrospective/influenza/scores_balanced.pickle'
confusionfile_balanced = '../data/retrospective/influenza/confusions_balanced.pickle'
featurefile_balanced = '../data/retrospective/influenza/features_balanced.pickle'

featoutfile = '../data/retrospective/influenza/key_features.txt'
featoutfile_balanced = '../data/retrospective/influenza/key_features_balanced.txt'

stdfeatoutfile = '../data/retrospective/influenza/gsym_key_features.txt'
stdfeatrankoutfile = '../data/retrospective/influenza/gsym_feature_ranks.csv'

#z = model(datafile, labelfile)
#z.save()
#z = pickle.load(open(modelfile))

#z = model(stddatafile)
#z.save()
#z = pickle.load(open(stdmodelfile))

p = rf_tests(stdmodelfile)
#p.score_all(stdscorefile, True)
#p.confusion_all(confusionfile, False)
#p.key_features(stdfeatoutfile, stdfeaturefile, False)
p.rank_features(stdfeatrankoutfile, stdfeaturefile, False)

#p_balanced = rf_tests(modelfile, 'balanced_subsample')
#p_balanced.score_all(scorefile_balanced, False)
#p_balanced.confusion_all(confusionfile_balanced, False)
#p_balanced.key_features(featoutfile_balanced, featurefile_balanced, False)