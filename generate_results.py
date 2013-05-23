import csv
import numpy as np
import stream_utils as streams
import early_stopping
import mlp, lstsq, logistic

def lstsq_interval_selection():
	print '-- Least Squares Regression ----------- :'
	splits = 10
	seed = 123456
	model_params = np.concatenate([[0.0], 0.05 * np.exp2(np.arange(0, 10))])
	model_testing_errors = np.empty(shape=(model_params.size, 1 + splits))
	for i,v in enumerate(model_params):
		print ' - Generating for param (v)             :', v
		classifier = lstsq.LeastSquares(v)
		errors = classerrors = []
		for s,(training_stream,validation_stream) in enumerate(streams.cross_validation_5class(splits=splits, seed=seed)):
			print '   -> cross validation                  :', s+1, '/', splits
			x_left, x_right, t = training_stream.all()
			classifier.train(x_left, x_right, t)
			error, classerror = classifier.normalized_error(validation_stream)
			errors = errors + [error]
		print '   average error                        :', float(sum(errors)) / float(len(errors))
		model_testing_errors[i] = [v] + errors
	np.savetxt('results/lstsq_interval.txt', model_testing_errors)


def lstsq_model_selection():
	print '-- Least Squares Regression ----------- :'
	splits = 10
	seed = 123456
	model_params = np.arange(0.0,25*5e-2,5e-2)
	model_testing_errors = np.empty(shape=(model_params.size, 1 + splits))
	model_testing_classerrors = np.empty(shape=(model_params.size, 1 + splits))
	for i,v in enumerate(model_params):
		print ' - Generating for param (v)             :', v
		classifier = lstsq.LeastSquares(v)
		errors = classerrors = []
		for s,(training_stream,validation_stream) in enumerate(streams.cross_validation_5class(splits=splits, seed=seed)):
			print '   -> cross validation                  :', s+1, '/', splits
			x_left, x_right, t = training_stream.all()
			classifier.train(x_left, x_right, t)
			error, classerror = classifier.normalized_error(validation_stream)
			errors, classerrors = errors + [error], classerrors + [classerror]
		print '   average error                        :', float(sum(errors)) / float(len(errors))
		print '   average classification errors        :', float(sum(classerrors)) / float(len(classerrors))
		model_testing_errors[i] = [v] + errors
		model_testing_classerrors[i] = [v] + classerrors
	np.savetxt('results/lstsq_errors.txt', model_testing_errors)
	np.savetxt('results/lstsq_classerrors.txt', model_testing_classerrors)

def cartesian(arrays):
	return np.dstack(np.meshgrid(*arrays)).reshape(-1, 2)

def cartesian3(nus,mus,counts):
	return np.array([(nu,mu,count) for count in counts for mu in mus for nu in nus])


nus_repr = ['0.01','0.02','0.05','1/x','1/10x','1/100x','1/sqrt(x)','1/10sqrt(x)','1/100sqrt(x)','1/cbrt(x)','1/10cbrt(x)','1/100cbrt(x)']
nus = np.array([
	(0, 0.01), (1, 0.02), (2, 0.05), (3, lambda x: 1.0/float(x)), (4, lambda x: 0.1/float(x)),
	(5, lambda x: 0.01/float(x)), (6, lambda x: 1.0/pow(float(x),0.5)), (7, lambda x: 0.1/pow(float(x),0.5)),
	(8, lambda x: 0.01/pow(float(x),0.5)), (9, lambda x: 1.0/pow(float(x),0.33)),
	(10, lambda x: 0.1/pow(float(x),0.33)), (11, lambda x: 0.01/pow(float(x),0.33))
])
# nus = np.array([
# 	('0.01', 0.01), ('0.02', 0.02), ('0.05', 0.05), ('1/x', lambda x: 1.0/float(x)), ('1/10x', lambda x: 0.1/float(x)),
# 	('1/100x', lambda x: 0.01/float(x)), ('1/sqrt(x)', lambda x: 1.0/pow(float(x),0.5)), ('1/10sqrt(x)', lambda x: 0.1/pow(float(x),0.5)),
# 	('1/100sqrt(x)', lambda x: 0.01/pow(float(x),0.5)), ('1/cbrt(x)', lambda x: 1.0/pow(float(x),0.33)),
# 	('1/10cbrt(x)', lambda x: 0.1/pow(float(x),0.33)), ('1/100cbrt(x)', lambda x: 0.01/pow(float(x),0.33))
# ])
mus = np.arange(0.05, 5*0.05, 0.05)
counts = np.array([1, 2, 5, 10, 20, 50])
gradient_model_params = [((nu_repr,nu),mu,count) for nu_repr,nu in nus for mu in mus for count in counts]

def get_results(fname, n=1):
	data = np.loadtxt(fname)
	data = data[np.invert(np.isnan(data[:,-2])),:]
	out = []
	for i in range(n):
		index = data[:,-2].argmin()
		out.append(data[index,:])
		data = np.delete(data, index, 0)
	return np.array(out)


def logistic_descent_model_selection():
	seed = 123541
	print '-- Logistic Regression ---------------- :'
	with open('results/logistic_descent.txt', 'w') as csvfile:
		writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for (nu_repr,nu),mu,count in gradient_model_params:
			print ' - Generating for params (nu,mu,count)  :', nu_repr, mu, count
			classifier = logistic.LogisticLoss(nu,mu)
			training, validation = streams.validation_5class(count=100, seed=seed)
			trained, chain, seconds = early_stopping.run(training, validation, classifier, count=count, max_time=600)
			error, classerror = trained.normalized_error(validation)
			print '   chain length                         :', len(chain)
			print '   seconds                              :', seconds
			print '   errors                               :', error
			print '   classification errors                :', classerror
			writer.writerow([nu_repr, mu, count, len(chain), seconds, error, classerror])

def logistic_descent():
	print '-- Logistic Regression ---------------- :'
	params = [
		[0.02, 0.05, 5],  # results/logistic_1.txt
		[0.02, 0.05, 10], # results/logistic_2.txt
		[0.02, 0.05, 20], # results/logistic_3.txt
		[0.02, 0.15, 2],  # results/logistic_4.txt
		[0.02, 0.2, 20]   # results/logistic_5.txt
	]
	for i,(nu,mu,count) in enumerate(params):
		with open('results/logistic_' + str(i + 1) + '.txt', 'w') as csvfile:
			writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			print ' - Generating for params (nu,mu,count)  :', nu, mu, count
			for _ in range(0, 10):
				classifier = logistic.LogisticLoss(nu,mu)
				training, validation = streams.validation_5class()
				trained, chain, seconds = early_stopping.run(training, validation, classifier, count=count, max_time=300)
				error, classerror = trained.normalized_error(validation)
				print '   chain length                         :', len(chain)
				print '   seconds                              :', seconds
				print '   errors                               :', error
				print '   classification errors                :', classerror
				row = [nu, mu, count, seconds, error, classerror] + chain
				writer.writerow(row)

def mlp_binary_descent_model_selection():
	seed = 2819732
	print '-- Binary MLP Gradient Descent -------- :'
	fname = 'results/mlp_binary_descent.txt'
	try:
		model_testing = np.loadtxt(fname)
	except IOError:
		model_testing = np.zeros(shape=(len(gradient_model_params), 7))
	for i,((nu_repr,nu),mu,count) in enumerate(gradient_model_params):
		if np.nanmax(model_testing[i,:]) > 0.0: continue
		print ' - Generating for params (nu,mu,count)  :', nu_repr, mu, count
		classifier = mlp.MLP(10, 10, nu=nu, mu=mu, k=2)
		training, validation = streams.validation_binary(seed=seed)
		trained, chain, seconds = early_stopping.run(training, validation, classifier, count=count)
		error, classerror = trained.normalized_error(validation)
		print '   chain length                         :', len(chain)
		print '   seconds                              :', seconds
		print '   errors                               :', error
		print '   classification errors                :', classerror
		model_testing[i] = [nu_repr, mu, count, len(chain), seconds, error, classerror]
		np.savetxt(fname, model_testing)
	nmin = 5
	best = get_results(fname, nmin)
	print "binary descent parameters:"
	for i in range(nmin):
		print "		error:", best[i,5]
		print "		nu:", nus_repr[int(best[i,0])]
		print "		mu:", best[i,1]
		print "		count:", best[i,2]

def mlp_5class_descent_model_selection():
	seed = 12398123
	print '-- 5Class MLP Gradient Descent -------- :'
	fname = 'results/mlp_5class_descent.txt'
	try:
		model_testing = np.loadtxt(fname)
	except IOError:
		model_testing = np.zeros(shape=(len(gradient_model_params), 7))
	for i,((nu_repr,nu),mu,count) in enumerate(gradient_model_params):
		if np.nanmax(model_testing[i,:]) > 0.0: continue
		print ' - Generating for params (nu,mu,count)  :', nu_repr, mu, count
		classifier = mlp.MLP(10, 10, nu=nu, mu=mu, k=5)
		training, validation = streams.validation_5class(seed=seed)
		trained, chain, seconds = early_stopping.run(training, validation, classifier, count=count)
		error, classerror = trained.normalized_error(validation)
		print '   chain length                         :', len(chain)
		print '   seconds                              :', seconds
		print '   errors                               :', error
		print '   classification errors                :', classerror
		model_testing[i] = [nu_repr, mu, count, len(chain), seconds, error, classerror]
		np.savetxt(fname, model_testing)
	best = get_results(fname,5)
	print "5class descent parameters:", best

def mlp_binary_model_selection():
	seed = 293482934
	print '-- Binary MLP ------------------------- :'
	H1s = H2s = np.arange(10,9*10,10)
	nus = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
	fname = 'results/mlp_binary.txt'
	model_params = cartesian3(nus, H1s, H2s)
	try:
		model_testing = np.loadtxt(fname)
	except IOError:
		model_testing = np.zeros(shape=(len(model_params), 5))
	for i,(nu,H1,H2) in enumerate(model_params):
		if np.nanmax(model_testing[i,:]) > 0.0: continue
		print ' - Generating for params (nu,H1,H2)        :', nu, H1, H2
		classifier = mlp.MLP(H1, H2, k=2, nu=nu, mu=0.1)
		training, validation = streams.validation_binary(seed=seed)
		trained, _, _ = early_stopping.run(training, validation, classifier, count=20, max_time=100)
		error, classerror = trained.normalized_error(validation)
		print '   errors                               :', error
		print '   classification errors                :', classerror
		model_testing[i] = [nu, H1, H2, error, classerror]
		np.savetxt(fname, model_testing)
	best = get_results(fname, 5)
	print "binary parameters:", best

def mlp_5class_model_selection():
	seed = 23427310
	print '-- 5Class MLP ------------------------- :'
	H1s = H2s = np.arange(10,9*10,10)
	nus = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
	fname = 'results/mlp_5class.txt'
	model_params = cartesian3(nus, H1s, H2s)
	try:
		model_testing = np.loadtxt(fname)
	except IOError:
		model_testing = np.zeros(shape=(len(model_params), 5))
	for i,(nu,H1,H2) in enumerate(model_params):
		if np.nanmax(model_testing[i,:]) > 0.0: continue
		print ' - Generating for params (nu,H1,H2)        :', nu, H1, H2
		classifier = mlp.MLP(H1, H2, k=5, nu=nu, mu=0.1)
		training, validation = streams.validation_5class(seed=seed)
		trained, _, _ = early_stopping.run(training, validation, classifier, count=5, max_time=100)
		error, classerror = trained.normalized_error(validation)
		print '   errors                               :', error
		print '   classification errors                :', classerror
		model_testing[i] = [nu, H1, H2, error, classerror]
		np.savetxt(fname, model_testing)
	best = get_results(fname,5)
	print "5class parameters:", best

# mlp_binary_descent_model_selection()
# mlp_5class_descent_model_selection()
# mlp_binary_model_selection()
mlp_5class_model_selection()
